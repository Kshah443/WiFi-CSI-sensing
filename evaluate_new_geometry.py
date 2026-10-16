#!/usr/bin/env python3
"""Independent antenna geometry experiment: train 06, hold out 07.

Parsing, amplitude conversion, windowing, features and RF configuration copied
unchanged from the original localization pipeline. Never saves a model.
"""
import argparse
import csv
import hashlib
import json
import logging
import platform
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = [f'P{i}' for i in range(1, 10)]
COORDINATES = {label: (-2 + 2 * (i % 3), 2 + 2 * (i // 3)) for i, label in enumerate(LABELS)}
FEATURE_STATS = ['mean', 'std', 'variance', 'min', 'max', 'median', 'q25', 'q75', 'rms', 'diff_abs_mean', 'diff_std']
PREPROCESSING_VERSION = '1.0'


def parse_csi_line(line):
    """Return raw signed integer I/Q; None for non-CSI lines; raise on corruption.

    No eval is used. ESP CSI samples are signed int8. Incomplete final lines,
    odd vectors, noninteger values and out-of-range samples are rejected.
    """
    if 'CSI_DATA' not in line:
        return None
    if not line.endswith('\n'):
        raise ValueError('incomplete_line')
    match = re.search(r'"\[([^\[\]]*)\]"\s*$', line)
    if not match:
        raise ValueError('missing_quoted_array')
    body = match.group(1).strip()
    if not re.fullmatch(r'[+-]?\d+(?:\s*,\s*[+-]?\d+)*', body):
        raise ValueError('invalid_integer_array')
    values = [int(x) for x in body.split(',')]
    if len(values) < 2 or len(values) % 2:
        raise ValueError('odd_or_empty_iq_length')
    if any(x < -128 or x > 127 for x in values):
        raise ValueError('sample_outside_int8')
    return np.asarray(values, dtype=np.float32)


def iq_to_amplitude(iq):
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 1 or len(iq) < 2 or len(iq) % 2 or not np.isfinite(iq).all():
        raise ValueError('invalid_iq_vector')
    return np.hypot(iq[0::2], iq[1::2])


def window_packets(packets, window_size=100):
    """Nonoverlapping full windows only; never pads a trial's trailing packets."""
    a = np.asarray(packets, dtype=np.float32)
    if window_size < 2 or a.ndim != 2:
        raise ValueError('Expected packet matrix and window_size >= 2')
    n = len(a) // window_size
    return a[:n * window_size].reshape(n, window_size, a.shape[1])


def feature_names(amplitude_length):
    return [f'{stat}_sc_{i}' for stat in FEATURE_STATS for i in range(amplitude_length)] + ['global_mean', 'global_std', 'global_rms']


def extract_features(window):
    """Stat-major features over every subcarrier, followed by three global stats.

    Raw amplitudes, no fitted normalization, no phase, no smoothing or AGC
    correction; real-time inference must use this exact function.
    """
    a = np.asarray(window, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 2 or not np.isfinite(a).all():
        raise ValueError('Invalid amplitude window')
    d = np.diff(a, axis=0)
    parts = [a.mean(0), a.std(0), a.var(0), a.min(0), a.max(0), np.median(a, axis=0),
             np.quantile(a, .25, axis=0), np.quantile(a, .75, axis=0),
             np.sqrt(np.mean(a*a, axis=0)), np.abs(d).mean(0), d.std(0)]
    return np.concatenate(parts + [np.array([a.mean(), a.std(), np.sqrt(np.mean(a*a))])]).astype(np.float32)


def labels_to_xy(labels):
    try:
        return np.asarray([COORDINATES[str(label)] for label in labels], dtype=float).reshape(-1, 2)
    except KeyError as exc:
        raise ValueError(f'Unknown position: {exc.args[0]}') from exc


def localization_errors(true_labels, predicted_labels):
    if len(true_labels) != len(predicted_labels):
        raise ValueError('Label counts differ')
    return np.linalg.norm(labels_to_xy(true_labels) - labels_to_xy(predicted_labels), axis=1)


def split_trials(trials, seed=42):
    rng = np.random.default_rng(seed)
    split = {'seed': seed, 'train': [], 'test': []}
    for label in LABELS:
        names = sorted(t['filename'] for t in trials if t['label'] == label)
        if len(names) != 5 or len(set(names)) != 5:
            raise ValueError(f'{label}: expected exactly five distinct trials')
        held_out = names[int(rng.integers(5))]
        split['test'].append(held_out)
        split['train'].extend(name for name in names if name != held_out)
    return split


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load_trials(root):
    trials = []
    for label in LABELS:
        for number in (6, 7):
            path = root / label / f'{label}_{number:02d}.txt'
            if not path.is_file():
                raise FileNotFoundError(f'Missing required trial: {path}')
            packets, reasons, examples = [], Counter(), []
            total = non_csi = 0
            with path.open(encoding='utf-8', errors='replace') as stream:
                for lineno, line in enumerate(stream, 1):
                    total += 1
                    try:
                        iq = parse_csi_line(line)
                        if iq is None:
                            non_csi += 1
                        else:
                            packets.append(iq_to_amplitude(iq))
                    except (ValueError, OverflowError) as exc:
                        reasons[str(exc)] += 1
                        if len(examples) < 10:
                            examples.append({'line': lineno, 'reason': str(exc)})
            trials.append({'label': label, 'filename': f'{label}/{path.name}', 'packets': packets,
                           'audit': {'total_lines': total, 'non_csi_lines': non_csi,
                                     'parsed_packets': len(packets), 'corrupt_packets': sum(reasons.values()),
                                     'corruption_reasons': dict(reasons), 'corruption_examples': examples,
                                     'amplitude_length_histogram': dict(Counter(map(len, packets))),
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}})
    return trials


def run(args):
    output = Path(args.output).expanduser()
    root = Path(args.data).expanduser().resolve()
    output = output.resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('Output must be separate from the dataset')
    # Exclusive directory creation: never overwrite an earlier experiment.
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output/'training.log', mode='w')], force=True)
    logging.info('Loading dataset %s', args.data)
    trials = load_trials(Path(args.data).expanduser())
    split = {'seed': args.seed,
             'train': [f'{p}/{p}_06.txt' for p in LABELS],
             'test': [f'{p}/{p}_07.txt' for p in LABELS]}
    logging.info('SMALL GEOMETRY EXPERIMENT: 1 training trial (06) and 1 completely held-out test trial (07) per position. Compare directionally only against the original setup; adjacent windows are correlated.')
    write_json(output/'trial_split.json', split)
    training_lengths = Counter(len(p) for t in trials if t['filename'] in split['train'] for p in t['packets'])
    if not training_lengths:
        raise ValueError('No valid training CSI packets')
    # Deterministic tie breaking, learned exclusively from training trials.
    vector_length = min(training_lengths, key=lambda n: (-training_lengths[n], n))
    logging.info('Keeping amplitude length %d (raw I/Q length %d); train-only mode', vector_length, vector_length*2)
    dataset = {'train': {'x': [], 'y': [], 'rows': []}, 'test': {'x': [], 'y': [], 'rows': []}}
    audit, unusable = {}, []
    typical = np.median([len(t['packets']) for t in trials if t['filename'] in split['train']])
    for t in trials:
        part = 'train' if t['filename'] in split['train'] else 'test'
        accepted = [p for p in t['packets'] if len(p) == vector_length]
        windows = window_packets(np.asarray(accepted, dtype=np.float32).reshape(-1, vector_length), args.window_size)
        a = t['audit']
        a.update(split=part, used_packets=len(windows)*args.window_size, accepted_packets=len(accepted),
                 rejected_length_packets=len(t['packets'])-len(accepted),
                 trailing_packets=len(accepted)%args.window_size, windows=len(windows))
        audit[t['filename']] = a
        if len(accepted) < max(args.window_size, .5*typical):
            logging.warning('%s: unusually few usable packets (%d); training median %.0f', t['filename'], len(accepted), typical)
        if a['corrupt_packets'] or a['rejected_length_packets']:
            logging.warning('%s: %d corrupt, %d length mismatches rejected', t['filename'], a['corrupt_packets'], a['rejected_length_packets'])
        if not len(windows):
            unusable.append(t['filename'])
        for i, w in enumerate(windows):
            dataset[part]['x'].append(extract_features(w))
            dataset[part]['y'].append(t['label'])
            dataset[part]['rows'].append({'trial_filename': t['filename'], 'window_index': i})
    write_json(output/'packet_audit.json', {'selected_amplitude_length': vector_length, 'selection_source': 'training_trials_only', 'trials': audit})
    for label in LABELS:
        for part in ('train', 'test'):
            group = [a for name, a in audit.items() if name.startswith(label+'/') and a['split'] == part]
            logging.info('%s %s: %d trials, %d accepted packets, %d used packets, %d windows', label, part,
                         sum(a['windows'] > 0 for a in group), sum(a['accepted_packets'] for a in group),
                         sum(a['used_packets'] for a in group), sum(a['windows'] for a in group))
    if unusable:
        raise ValueError('Trials without a full window; refusing incomplete 1/1 geometry evaluation: '+', '.join(unusable))
    train, test = dataset['train'], dataset['test']
    model = RandomForestClassifier(n_estimators=args.trees, random_state=args.seed, n_jobs=args.jobs,
                                   class_weight='balanced_subsample', max_features='sqrt')
    model.fit(np.asarray(train['x']), train['y'])
    pred = model.predict(np.asarray(test['x']))
    probabilities = model.predict_proba(np.asarray(test['x']))
    errors = localization_errors(test['y'], pred)
    cm = confusion_matrix(test['y'], pred, labels=LABELS)
    pairs = sorted([{'true_label': LABELS[i], 'predicted_label': LABELS[j], 'count': int(cm[i,j])}
                    for i in range(9) for j in range(9) if i != j and cm[i,j]], key=lambda p: -p['count'])
    metrics = {'training_accuracy': float(accuracy_score(train['y'], model.predict(np.asarray(train['x'])))),
               'test_accuracy': float(accuracy_score(test['y'], pred)), 'exact_position_accuracy': float(np.mean(errors == 0)),
               'accuracy_within_2ft': float(np.mean(errors <= 2 + 1e-9)),
               'mean_error_ft': float(errors.mean()), 'median_error_ft': float(np.median(errors)),
               'p90_error_ft': float(np.percentile(errors,90)), 'worst_error_ft': float(errors.max()),
               'confusion_matrix_labels': LABELS, 'confusion_matrix': cm.tolist(), 'most_confused_pairs': pairs,
               'classification_report': classification_report(test['y'], pred, labels=LABELS, output_dict=True, zero_division=0),
               'train_windows': len(train['y']), 'test_windows': len(test['y']), 'train_trials': 9, 'test_trials': 9,
               'position_usage': {label: {part: {'trials': sum(a['split']==part for name,a in audit.items() if name.startswith(label+'/')),
                                              'windows': sum(a['windows'] for name,a in audit.items() if name.startswith(label+'/') and a['split']==part)}
                                         for part in ('train','test')} for label in LABELS}}
    metrics['experiment'] = 'Small 1-trial-train / 1-trial-test geometry experiment; compare directionally only against original setup.'
    metrics['original_benchmark_reference'] = {'test_accuracy': 0.5157, 'mean_error_ft': 1.311, 'accuracy_within_2ft': 0.7547}
    write_json(output/'metrics.json', metrics)
    xy_true, xy_pred = labels_to_xy(test['y']), labels_to_xy(pred)
    fields = ['trial_filename', 'window_index', 'true_label', 'predicted_label', 'true_x', 'true_y', 'predicted_x', 'predicted_y', 'localization_error', 'prediction_confidence']
    with (output/'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(test['rows']):
            writer.writerow(dict(row, true_label=test['y'][i], predicted_label=pred[i], true_x=xy_true[i,0], true_y=xy_true[i,1],
                                 predicted_x=xy_pred[i,0], predicted_y=xy_pred[i,1], localization_error=errors[i], prediction_confidence=probabilities[i].max()))
    metadata = {'preprocessing_version': PREPROCESSING_VERSION, 'window_size_packets': args.window_size,
                'raw_iq_length': vector_length*2, 'amplitude_length': vector_length, 'feature_names': feature_names(vector_length),
                'feature_count': len(feature_names(vector_length)), 'feature_order': FEATURE_STATS,
                'coordinates_ft': COORDINATES, 'class_order': model.classes_.tolist(), 'seed': args.seed,
                'model_parameters': model.get_params(), 'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__,
                'python_version': platform.python_version(), 'joblib_version': joblib.__version__,
                'iq_format': 'alternating signed int8 components; hypot of each pair', 'std_ddof': 0,
                'quantile_method': 'linear', 'dtype': 'float32', 'normalization': 'none', 'subcarrier_mask': 'all',
                'window_stride_packets': args.window_size, 'trailing_packets': 'discard',
                'packet_policy': 'reject malformed/incomplete lines and length mismatches; concatenate accepted packets in original order',
                'confidence': 'maximum random forest class probability; uncalibrated',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output/'model_metadata.json', metadata)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(9,8))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Geometry experiment: train 06 / test 07 (counts)')
    fig.tight_layout()
    fig.savefig(output/'confusion_matrix.png', dpi=160)
    plt.close(fig)
    logging.info('Training accuracy %.2f%%; test accuracy %.2f%%', metrics['training_accuracy']*100, metrics['test_accuracy']*100)
    logging.info('\n%s', classification_report(test['y'], pred, labels=LABELS, zero_division=0))
    logging.info('Confusion matrix (rows true, columns predicted; P1–P9):\n%s', cm)
    logging.info('Error ft: mean %.3f; median %.3f; p90 %.3f; worst %.3f', errors.mean(), np.median(errors), np.percentile(errors,90), errors.max())
    logging.info('Exact %.2f%%; within 2 ft %.2f%%', metrics['exact_position_accuracy']*100, metrics['accuracy_within_2ft']*100)
    logging.info('Most confused directed pairs: %s', pairs[:5] or 'none')
    logging.info('Small 1-trial-train / 1-trial-test geometry experiment; compare directionally only against original setup. Adjacent test windows are correlated; this is stationary fixed-grid localization.')
    logging.info('Per-file packet audit (parsed includes length mismatches):')
    for filename, row in audit.items():
        logging.info('%s [%s]: lines=%d non-CSI=%d parsed=%d corrupt=%d accepted=%d length-rejected=%d used=%d trailing=%d windows=%d',
                     filename, row['split'], row['total_lines'], row['non_csi_lines'], row['parsed_packets'], row['corrupt_packets'],
                     row['accepted_packets'], row['rejected_length_packets'], row['used_packets'], row['trailing_packets'], row['windows'])
    logging.info('Outputs: %s. No model file was read, written or altered.', output)
    print('Original benchmark (reference only): test accuracy: 51.57%; mean localization error: 1.311 ft; within 2 ft: 75.47%', flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='/home/komail-shah/csi_localization/')
    parser.add_argument('--output', default='/home/komail-shah/csi_localization_geometry_test/')
    args = parser.parse_args()
    args.window_size, args.seed, args.trees, args.jobs = 100, 42, 400, -1
    try:
        run(args)
    except Exception:
        logging.exception('Training failed')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
