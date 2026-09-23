#!/usr/bin/env python3
"""Train an ESP32 CSI classifier with a reproducible, whole-trial holdout.

Run: python3 train_csi.py
Dependencies: numpy scikit-learn matplotlib joblib
Windows contain 100 consecutive *retained* packets by default. Corrupt and
nonmodal-length packets are removed first, so windows may span capture gaps.
No window ever spans files; incomplete final windows are discarded.
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import re
import sys

try:
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        ConfusionMatrixDisplay, accuracy_score, classification_report,
        confusion_matrix,
    )
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}. Install with:\n"
        "  python3 -m pip install numpy scikit-learn matplotlib joblib"
    ) from exc


LABELS = ("empty", "still", "walking")
ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]")
MARKER = re.compile(r"(?<![A-Za-z0-9_])CSI_DATA\s*,")
FINAL_ARRAY = re.compile(r'''[,]\s*(["'])(\[[^\[\]\r\n]*\])\1[ \t\r\n]*\Z''')
INTEGER_LIST = re.compile(r"\[[ \t]*[+-]?[0-9]+(?:[ \t]*,[ \t]*[+-]?[0-9]+)*[ \t]*,?[ \t]*\]")
BASE_STATS = ("mean", "std", "variance", "min", "max", "median")
DIFF_STATS = ("diff_mean", "diff_std", "diff_abs_mean", "diff_abs_max")
LOG = logging.getLogger("csi_training")


@dataclass
class Trial:
    path: Path
    label: str
    packets: list = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    features: object = None


def parse_packet(line):
    """Validate a complete record and return float64 amplitudes in array order.

    Accept signed decimal integers, including gain-compensated int16 samples.
    Validate the entire list before conversion; no eval or partial parsing.
    Magnitude is identical whether firmware emits I/Q or Q/I pairs.
    """
    line = ANSI.sub("", line)
    if "\ufffd" in line or "\x00" in line:
        raise ValueError("damaged text (invalid UTF-8 or NUL)")
    markers = list(MARKER.finditer(line))
    if len(markers) != 1:
        raise ValueError("missing or multiple CSI_DATA records")
    match = FINAL_ARRAY.search(line[markers[0].start():])
    if match is None:
        raise ValueError("missing complete final quoted CSI list")
    literal = match.group(2)
    if len(literal) > 100_000:
        raise ValueError("CSI list exceeds 100000-character safety limit")
    if INTEGER_LIST.fullmatch(literal) is None:
        raise ValueError("CSI list is empty or contains invalid integer tokens")
    tokens = literal[1:-1].strip().rstrip(",").split(",")
    if len(tokens) % 2:
        raise ValueError("odd number of I/Q values")
    # The ESP32 firmware uses signed 16-bit samples after gain compensation.
    # Range checking also rejects corrupt, implausibly large numeric tokens.
    try:
        values = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("invalid or excessively long integer token") from exc
    if any(value < -32768 or value > 32767 for value in values):
        raise ValueError("I/Q sample outside signed 16-bit range")
    iq = np.asarray(values, dtype=np.float64)
    return np.hypot(iq[0::2], iq[1::2])


def read_trial(path, label):
    trial = Trial(path.resolve(), label)
    audit = trial.audit = {
        "path": str(trial.path), "label": label, "lines": 0,
        "non_csi_lines": 0, "damaged_non_csi_lines": 0,
        "candidate_packets": 0, "valid_packets": 0, "corrupt_packets": 0,
        "errors": {}, "error_line_examples": {}, "length_counts_iq_values": {},
        "file_error": None, "skip_reason": None,
    }
    errors, lengths = Counter(), Counter()
    try:
        # Replacement exposes corruption instead of silently deleting bytes.
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                audit["lines"] += 1
                if "CSI_DATA" not in line:
                    audit["non_csi_lines"] += 1
                    audit["damaged_non_csi_lines"] += int("\ufffd" in line or "\x00" in line)
                    continue
                audit["candidate_packets"] += 1
                try:
                    amplitude = parse_packet(line)
                except ValueError as exc:
                    reason = str(exc)
                    errors[reason] += 1
                    examples = audit["error_line_examples"].setdefault(reason, [])
                    if len(examples) < 5:
                        examples.append(number)
                    continue
                trial.packets.append(amplitude)
                lengths[2 * amplitude.size] += 1
    except OSError as exc:
        # Never train on a partially read file.
        audit["file_error"] = str(exc)
        audit["skip_reason"] = "file could not be read completely"
        trial.packets.clear()
    audit["valid_packets"] = sum(lengths.values())
    audit["corrupt_packets"] = sum(errors.values())
    audit["errors"] = dict(errors)
    audit["length_counts_iq_values"] = dict(sorted(lengths.items()))
    if not trial.packets and audit["skip_reason"] is None:
        audit["skip_reason"] = "no valid CSI packets"
    LOG.info(
        "%s/%s: candidates=%d valid=%d corrupt=%d non-CSI=%d (damaged=%d)",
        label, path.name, audit["candidate_packets"], audit["valid_packets"],
        audit["corrupt_packets"], audit["non_csi_lines"], audit["damaged_non_csi_lines"],
    )
    for reason, count in errors.items():
        LOG.warning("  Skipped %d packet(s): %s; example lines %s", count, reason,
                    audit["error_line_examples"][reason])
    if audit["skip_reason"]:
        LOG.warning("  SKIPPED FILE: %s%s", audit["skip_reason"],
                    f": {audit['file_error']}" if audit["file_error"] else "")
    return trial


def extract_features(window, temporal=True):
    """One row, ordered by statistic, then by subcarrier; population std/var."""
    if window.ndim != 2 or window.shape[0] < 2 or window.shape[1] == 0:
        raise ValueError("Expected a window with >=2 packets and >=1 subcarrier")
    values = [
        window.mean(axis=0), window.std(axis=0), window.var(axis=0),
        window.min(axis=0), window.max(axis=0), np.median(window, axis=0),
    ]
    if temporal:
        difference = np.diff(window, axis=0)
        values.extend([
            difference.mean(axis=0), difference.std(axis=0),
            np.abs(difference).mean(axis=0), np.abs(difference).max(axis=0),
        ])
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite extracted feature")
    return result


def build_windows(trials, iq_length, window_size, temporal):
    usable = []
    for trial in trials:
        audit = trial.audit
        kept = [packet for packet in trial.packets if 2 * packet.size == iq_length]
        audit["length_mismatch_packets"] = len(trial.packets) - len(kept)
        audit["retained_packets"] = len(kept)
        audit["windows"] = len(kept) // window_size
        audit["trailing_packets_discarded"] = len(kept) % window_size
        if kept and audit["windows"]:
            trial.features = np.vstack([
                extract_features(np.stack(kept[start:start + window_size]), temporal)
                for start in range(0, len(kept) - window_size + 1, window_size)
            ])
            usable.append(trial)
        elif audit["skip_reason"] is None:
            audit["skip_reason"] = f"fewer than {window_size} packets of modal length"
        LOG.info("%s/%s: retained=%d length-skipped=%d windows=%d tail-discarded=%d%s",
                 trial.label, trial.path.name, len(kept), audit["length_mismatch_packets"],
                 audit["windows"], audit["trailing_packets_discarded"],
                 f"; SKIPPED FILE: {audit['skip_reason']}" if audit["skip_reason"] else "")
        trial.packets.clear()  # Release packet arrays once features are extracted.
    return usable


def split_trials(trials, test_size, seed):
    """Shuffle trial files within each class, keeping every class on both sides."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in LABELS:
        group = sorted((trial for trial in trials if trial.label == label), key=lambda t: str(t.path))
        if len(group) < 2:
            raise ValueError(
                f"Class '{label}' has {len(group)} usable trial(s); at least two are "
                "required for a whole-trial train/test split. Record more trials."
            )
        # Per-class ceiling, capped to leave at least one training trial.
        n_test = min(len(group) - 1, math.ceil(len(group) * test_size))
        order = rng.permutation(len(group))
        test.extend(group[i] for i in order[:n_test])
        train.extend(group[i] for i in order[n_test:])
    train_paths, test_paths = {t.path for t in train}, {t.path for t in test}
    if train_paths & test_paths:
        raise ValueError("Trial leakage detected: training and test files overlap")
    return train, test


def assemble(trials):
    return (np.vstack([trial.features for trial in trials]),
            np.concatenate([np.full(len(trial.features), trial.label) for trial in trials]))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    trials = []
    # The dataset-wide mode is a format compatibility filter, not a learned
    # statistic. No scaling, feature selection, or model fitting uses test data.
    lengths = Counter()
    seen_files = set()
    for label in LABELS:
        directory = args.dataset / label
        paths = sorted(directory.glob("*.txt"))
        if not paths:
            LOG.warning("No .txt trial files found in %s", directory)
        for path in paths:
            # Reject aliases/hard links so a physical recording cannot be used
            # twice, potentially with contradictory folder labels.
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
            except OSError:
                identity = None  # read_trial will report the specific I/O error.
            if identity is not None and identity in seen_files:
                raise ValueError(f"Duplicate physical trial file: {path}")
            if identity is not None:
                seen_files.add(identity)
            trial = read_trial(path, label)
            trials.append(trial)
            lengths.update(2 * packet.size for packet in trial.packets)
    audit_path = args.output / "packet_audit.json"
    audit = {"trials": [trial.audit for trial in trials],
             "length_counts_iq_values": dict(sorted(lengths.items()))}
    write_json(audit_path, audit)
    if not lengths:
        raise ValueError("No valid CSI packets in readable trial files. See packet_audit.json.")
    # Deterministic tie break: choose the smaller I/Q vector length.
    iq_length = min(lengths, key=lambda length: (-lengths[length], length))
    LOG.info("Dataset I/Q vector lengths (scalar count -> packets): %s", dict(sorted(lengths.items())))
    LOG.info("Keeping modal length: %d I/Q values = %d subcarriers", iq_length, iq_length // 2)
    usable = build_windows(trials, iq_length, args.window_size, not args.no_temporal)
    audit["selected_iq_length"] = iq_length
    audit["totals"] = {
        key: sum(trial.audit[key] for trial in trials)
        for key in ("candidate_packets", "valid_packets", "corrupt_packets", "non_csi_lines",
                    "damaged_non_csi_lines", "length_mismatch_packets", "retained_packets",
                    "trailing_packets_discarded", "windows")
    }
    audit["totals"]["files_found"] = len(trials)
    audit["totals"]["files_used"] = len(usable)
    audit["totals"]["files_skipped"] = len(trials) - len(usable)
    write_json(audit_path, audit)
    LOG.info("Dataset totals: %s", audit["totals"])
    train_trials, test_trials = split_trials(usable, args.test_size, args.seed)
    x_train, y_train = assemble(train_trials)
    x_test, y_test = assemble(test_trials)
    manifest = {}
    for name, group, labels in (("train", train_trials, y_train), ("test", test_trials, y_test)):
        manifest[name] = [
            {"path": str(t.path), "label": t.label, "windows": len(t.features)} for t in group
        ]
        LOG.info("%s: %d trial files, %d windows; windows per class: %s",
                 name.upper(), len(group), len(labels), dict(Counter(labels.tolist())))
        for label in LABELS:
            LOG.info("  %s files: %s", label, ", ".join(t.path.name for t in group if t.label == label))
    write_json(args.output / "trial_split.json", manifest)
    LOG.info("Verified: zero shared trial files between train and test.")
    LOG.info("Training RandomForest: %d trees, %d features/window", args.n_estimators, x_train.shape[1])
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.seed,
        class_weight="balanced", n_jobs=args.n_jobs,
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    train_accuracy = accuracy_score(y_train, model.predict(x_train))
    test_accuracy = accuracy_score(y_test, predicted)
    report = classification_report(y_test, predicted, labels=LABELS, digits=4, zero_division=0)
    matrix = confusion_matrix(y_test, predicted, labels=LABELS)
    LOG.info("Training accuracy: %.4f", train_accuracy)
    LOG.info("Held-out trial window accuracy: %.4f", test_accuracy)
    LOG.info("Classification report:\n%s", report)
    LOG.info("Confusion matrix (rows=true, columns=predicted; order=%s):\n%s", LABELS, matrix)
    figure, axes = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(
        ax=axes, cmap="Blues", values_format="d", colorbar=False,
    )
    axes.set_title(f"CSI: held-out trials (accuracy {test_accuracy:.1%})")
    figure.savefig(args.output / "confusion_matrix.png", dpi=200)
    plt.close(figure)
    joblib.dump(model, args.output / "random_forest.joblib", compress=3)
    stats = BASE_STATS + (() if args.no_temporal else DIFF_STATS)
    metadata = {
        "dataset": str(args.dataset.resolve()), "labels": list(LABELS),
        "iq_length": iq_length, "subcarriers": iq_length // 2,
        "window_size": args.window_size, "temporal_features": not args.no_temporal,
        "feature_order": [f"{name}_sc_{i}" for name in stats for i in range(iq_length // 2)],
        "std_variance_ddof": 0, "seed": args.seed, "test_fraction_requested": args.test_size,
        "split_rule": "per-class shuffled whole trials; ceil(n*fraction), capped at n-1",
        "window_rule": "non-overlapping retained packets within each file; drop trailing remainder",
        "amplitude_rule": "hypot(pair[0], pair[1]); preserve raw subcarrier order",
        "n_estimators": args.n_estimators,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    write_json(args.output / "model_metadata.json", metadata)
    write_json(args.output / "metrics.json", {
        "train_trials": len(train_trials), "test_trials": len(test_trials),
        "train_windows": len(y_train), "test_windows": len(y_test),
        "train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
        "classification_report": classification_report(
            y_test, predicted, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": matrix.tolist(), "confusion_matrix_labels": list(LABELS),
    })
    LOG.info("Saved model, confusion matrix, metrics, metadata, trial split, packet audit, and training.log to %s",
             args.output.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/home/komail-shah/csi_dataset"))
    parser.add_argument("--output", type=Path, default=Path("/home/komail-shah/csi_ml_results"))
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of trials per class (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal-difference features")
    args = parser.parse_args(argv)
    if args.window_size < 2 or not 0 < args.test_size < 1:
        parser.error("--window-size must be >=2 and --test-size must be between 0 and 1")
    if args.n_estimators < 1 or args.n_jobs == 0 or args.seed < 0 or args.seed > 2**32 - 1:
        parser.error("Use positive --n-estimators, nonzero --n-jobs, and --seed in [0, 2**32-1]")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output / "training.log", mode="w", encoding="utf-8"),
        ], force=True)
        run(args)
    except (OSError, ValueError) as exc:
        LOG.error("Training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
