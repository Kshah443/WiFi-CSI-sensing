#!/usr/bin/env python3
"""Plot raw ESP-CSI amplitudes; defaults match this capture's paths."""

import argparse
import ast
from collections import Counter
from pathlib import Path
import re
import sys
import warnings


# Strip complete ANSI CSI sequences (including colors) and OSC strings.
ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]")
MARKER = re.compile(r"(?<![A-Za-z0-9_])CSI_DATA\s*,")
FINAL_ARRAY = re.compile(r''',\s*(["'])(\[[^\[\]\r\n]*\])\1[ \t\r\n\x00-\x1f\x7f]*\Z''')
# Restrict literal_eval to a flat list of decimal integers, with bounded size.
INTEGER_LIST = re.compile(r"\[[ \t]*[+-]?[0-9]+(?:[ \t]*,[ \t]*[+-]?[0-9]+)*[ \t]*,?[ \t]*\]")


def parse_packet(line):
    """Return one flat, nonempty, even-length integer list or raise ValueError."""
    line = ANSI.sub("", line)
    markers = list(MARKER.finditer(line))
    if len(markers) != 1:
        raise ValueError("missing or multiple CSI_DATA records")
    record = line[markers[0].start():]
    match = FINAL_ARRAY.search(record)
    if match is None:
        raise ValueError("missing complete final quoted CSI array")
    literal = match.group(2)
    if len(literal) > 100_000 or INTEGER_LIST.fullmatch(literal) is None:
        raise ValueError("CSI array is corrupt or is not a flat integer list")
    try:
        values = ast.literal_eval(literal)  # Never use eval on serial input.
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("invalid CSI integer list") from exc
    if not values or len(values) % 2:
        raise ValueError("CSI array must contain complete I/Q pairs")
    # The local ESP-CSI firmware emits int16_t values after gain compensation.
    if any(type(value) is not int or not -32768 <= value <= 32767 for value in values):
        raise ValueError("CSI sample is outside the firmware's signed 16-bit range")
    return values


def read_packets(path):
    packets = []
    candidates = 0
    errors = Counter()
    # Replacement characters remain in damaged arrays so they fail validation.
    # Ignoring undecodable bytes could silently change a numeric value.
    with path.open("r", encoding="utf-8", errors="replace") as capture:
        for line in capture:
            if "CSI_DATA" not in line:
                continue
            candidates += 1
            try:
                packets.append(parse_packet(line))
            except ValueError as exc:
                errors[str(exc)] += 1

    print(f"CSI_DATA lines: {candidates}")
    print(f"Successfully parsed CSI packets: {len(packets)}")
    print(f"Malformed CSI lines skipped: {sum(errors.values())}")
    for reason, count in errors.items():
        print(f"  {count}: {reason}")
    if not packets:
        raise ValueError(
            f"No valid CSI packets found in {path}. Expected CSI_DATA records "
            'ending in a quoted, nonempty, even-length integer list: "[0,2,-1,3]". '
            "Check for truncated records, serial corruption, and capture settings."
        )

    lengths = Counter(map(len, packets))
    common_length, _ = lengths.most_common(1)[0]  # Ties: first encountered length.
    kept = [packet for packet in packets if len(packet) == common_length]
    print(f"Packet lengths (scalar values -> count): {dict(lengths)}")
    print(f"Most common length: {common_length} values ({common_length // 2} I/Q pairs)")
    print(f"Inconsistent-length packets discarded: {len(packets) - len(kept)}")
    print(f"Packets used in heatmap: {len(kept)}")
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/home/komail-shah/csi_raw_test.txt"))
    parser.add_argument("--output", type=Path, default=Path("/home/komail-shah/csi_heatmap.png"))
    parser.add_argument("--no-show", action="store_true", help="Save without opening a GUI window")
    args = parser.parse_args()

    try:
        packets = read_packets(args.input)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        import numpy as np
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        print(f"Error: {exc}. Install NumPy and Matplotlib in this Python environment.", file=sys.stderr)
        return 1

    # Float conversion before arithmetic avoids overflow in integer squaring.
    data = np.asarray(packets, dtype=np.float64)
    # Espressif stores imaginary (Q) first, then real (I).
    Q = data[:, 0::2]
    I = data[:, 1::2]
    amplitude = np.hypot(I, Q)  # sqrt(I**2 + Q**2)

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    heatmap = ax.imshow(
        amplitude.T, origin="lower", aspect="auto", interpolation="nearest",
        cmap="viridis",
    )
    ax.set_xlabel("Retained packet index (capture order)")
    ax.set_ylabel("Subcarrier index (raw array order)")
    ax.set_title(f"ESP32-C5 CSI amplitude | {data.shape[0]} packets, {I.shape[1]} subcarriers")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.colorbar(heatmap, ax=ax, label="Amplitude (raw units)")
    try:
        fig.savefig(args.output, dpi=200)
    except OSError as exc:
        print(f"Error saving plot to {args.output}: {exc}", file=sys.stderr)
        plt.close(fig)
        return 1
    print(f"Saved heatmap to {args.output}", flush=True)

    if not args.no_show:
        try:
            # Turn a noninteractive-backend warning into a clear diagnostic.
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=".*non-interactive.*")
                plt.show(block=True)
        except Exception as exc:
            print(f"Plot saved, but interactive display failed: {exc}. "
                  "Run from a desktop session with a Matplotlib GUI backend.", file=sys.stderr)
            plt.close(fig)
            return 1
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
