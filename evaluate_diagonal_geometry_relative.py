#!/usr/bin/env python3
"""Evaluate diagonal geometry using trials 01 and 02 only.
Standalone; never loads or saves a trained model or edits input recordings.
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
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = [f'P{i}' for i in range(1, 10)]
COORDINATES = {label: (5 - 2 * (i % 3), 1 + 3 * (i // 3)) for i, label in enumerate(LABELS)}
FEATURE_STATS = ['mean', 'std', 'variance', 'min', 'max', 'median', 'q25', 'q75', 'rms', 'diff_abs_mean', 'diff_std']
EPS = 1e-8
BINS = 117
WINDOW = 100
RF_PARAMETERS = dict(n_estimators=400, random_state=42, n_jobs=-1,
                     class_weight='balanced_subsample', max_features='sqrt')
FORMULAS = {
 'raw': 'A[p,b] = hypot(I[p,b], Q[p,b]); unchanged amplitudes.',
 'packet_mean': 'A[p,b] / max(mean_b(A[p,b]), epsilon).',
 'packet_l2': 'A[p,b] / max(sqrt(sum_b(A[p,b]^2)), epsilon).',
 'packet_zscore': '(A[p,b] - mean_b(A[p,b])) / max(std_b(A[p,b], ddof=0), epsilon).',
 'window_mean': 'A[p,b] / max(mean_{p,b}(A[p,b]), epsilon), independently for each 100-packet window.',
 'relative_shape': 'Let g=max(mean_{p,b}(A),epsilon), mu[b]=mean_p(A), s[b]=std_p(A,ddof=0), d=diff_p(A). Concatenate mu/g, s/g, s/max(mu,epsilon), mean_p(abs(d))/g in that order. No global magnitude features.'
}

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

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def variant_features(window, variant):
    a = np.asarray(window, dtype=np.float32)
    if variant == 'relative_shape':
        g = max(float(a.mean()), EPS)
        mu, sigma = a.mean(0), a.std(0)
        return np.concatenate([mu/g, sigma/g, sigma/np.maximum(mu, EPS),
                               np.abs(np.diff(a, axis=0)).mean(0)/g]).astype(np.float32)
    if variant == 'packet_mean':
        a = a / np.maximum(a.mean(1, keepdims=True), EPS)
    elif variant == 'packet_l2':
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), EPS)
    elif variant == 'packet_zscore':
        a = (a-a.mean(1, keepdims=True)) / np.maximum(a.std(1, keepdims=True), EPS)
    elif variant == 'window_mean':
        a = a / max(float(a.mean()), EPS)
    elif variant != 'raw':
        raise ValueError(variant)
    return extract_features(a)


def variant_names(variant):
    if variant == 'relative_shape':
        return [f'{stat}_sc_{b}' for stat in ('mean_shape', 'relative_std', 'temporal_cv', 'relative_diff_abs_mean') for b in range(BINS)]
    return feature_names(BINS)


def load_data(root):
    windows, audits = {}, {}
    for label in LABELS:
        for number in (1, 2):
            name = f'{label}/{label}_{number:02d}.txt'
            path = root/name
            packets, reasons = [], Counter()
            examples, lengths = [], Counter()
            total = non_csi = parsed = rejected = 0
            with path.open(encoding='utf-8', errors='replace') as stream:
                for line_number, line in enumerate(stream, 1):
                    total += 1
                    try:
                        iq = parse_csi_line(line)
                        if iq is None:
                            non_csi += 1
                            continue
                        amplitude = iq_to_amplitude(iq)
                        parsed += 1
                        lengths[len(amplitude)] += 1
                        if len(amplitude) != BINS:
                            rejected += 1
                            continue
                        packets.append(amplitude)
                    except (ValueError, OverflowError) as exc:
                        reasons[str(exc)] += 1
                        if len(examples) < 10:
                            examples.append({'line': line_number, 'reason': str(exc)})
            w = window_packets(np.asarray(packets, dtype=np.float32).reshape(-1, BINS), WINDOW)
            audits[name] = dict(total_lines=total, non_csi_lines=non_csi, parsed_packets=parsed,
                                corrupt_packets=sum(reasons.values()), corruption_reasons=dict(reasons),
                                corruption_examples=examples, amplitude_length_histogram=dict(lengths),
                                rejected_length_packets=rejected, accepted_packets=len(packets),
                                used_packets=len(w)*WINDOW, trailing_packets=len(packets)%WINDOW,
                                windows=len(w), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            windows[(label, number)] = w
    return windows, audits


def scores(y, pred):
    errors = localization_errors(y, pred)
    return dict(exact_position_accuracy=float(accuracy_score(y, pred)),
                mean_error_ft=float(errors.mean()), median_error_ft=float(np.median(errors)),
                accuracy_within_2ft=float(np.mean(errors <= 2+1e-9)))




def weighted_xy(probabilities, class_order):
    """Map probability columns using model.classes_, never assumed label order."""
    probabilities = np.asarray(probabilities)
    return probabilities @ labels_to_xy(class_order)


def run(args):
    from datetime import datetime, timezone
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Output must be separate from the dataset')
    output.mkdir(parents=True, exist_ok=False)
    windows, audit = load_data(root)
    write_json(output/'packet_audit.json', audit)
    empty = [str(k) for k, w in windows.items() if not len(w)]
    if empty:
        raise ValueError('No full 100-packet windows: ' + ', '.join(empty))
    features = {k:np.asarray([variant_features(w, 'relative_shape') for w in ws]) for k,ws in windows.items()}
    if any(x.shape[1] != 468 or not np.isfinite(x).all() for x in features.values()):
        raise ValueError('Invalid relative_shape features')
    manifest = dict(geometry_id='diagonal_6x8_bottom_left', area_ft=[6,8],
        origin='bottom-left', coordinates_ft=COORDINATES, class_labels=LABELS,
        dataset_root=str(root), trials=['01','02'], raw_iq_length=234,
        amplitude_length=BINS, window_size_packets=WINDOW, window_stride_packets=WINDOW,
        preprocessing_variant='relative_shape', formula=FORMULAS['relative_shape'],
        epsilon=EPS, std_ddof=0, dtype='float32', feature_names=variant_names('relative_shape'),
        feature_count=468, random_forest_parameters=RF_PARAMETERS,
        localization_error='Euclidean distance in feet between true and predicted class coordinates',
        weighted_xy='predict_proba @ coordinates in model.classes_ order',
        temporal_inference='Offline independent windows; confidence/EMA/UDP belong to unchanged live pipeline and are not applied here.',
        versions=dict(numpy=np.__version__, sklearn=sklearn.__version__, python=platform.python_version()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'experiment_manifest.json', manifest)
    folds, pooled_true, pooled_pred = [], [], []
    prediction_rows = []
    fig, axes = plt.subplots(1,3,figsize=(16,5), constrained_layout=True)
    ax=axes[0]
    for p,(x,y) in COORDINATES.items():
        ax.scatter(x,y); ax.annotate(p,(x,y),xytext=(5,5),textcoords='offset points')
    ax.set(xlim=(0,6),ylim=(0,8),xlabel='X (ft)',ylabel='Y (ft)',title='6 × 8 ft; bottom-left origin')
    ax.set_aspect('equal'); ax.grid(True)
    for index,(name,train,test) in enumerate([('A',1,2),('B',2,1)]):
        train_keys=[(p,train) for p in LABELS]; test_keys=[(p,test) for p in LABELS]
        def stack(keys):
            return (np.concatenate([features[k] for k in keys]),
                    np.concatenate([np.repeat(k[0],len(features[k])) for k in keys]))
        x_train,y_train=stack(train_keys); x_test,y_test=stack(test_keys)
        model=RandomForestClassifier(**RF_PARAMETERS)
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
        probabilities=model.predict_proba(x_test)
        xy=weighted_xy(probabilities, model.classes_)
        cm=confusion_matrix(y_test,pred,labels=LABELS)
        recall={p:float(cm[i,i]/cm[i].sum()) for i,p in enumerate(LABELS)}
        fold=dict(fold=name,train_trial=f'{train:02d}',test_trial=f'{test:02d}',
            train_files=[f'{p}/{p}_{train:02d}.txt' for p in LABELS],
            test_files=[f'{p}/{p}_{test:02d}.txt' for p in LABELS],
            train_windows=len(y_train),test_windows=len(y_test),
            train_accuracy=float(accuracy_score(y_train,model.predict(x_train))),
            **scores(y_test,pred),confusion_matrix=cm.tolist(),per_position_recall=recall)
        folds.append(fold); pooled_true.extend(y_test); pooled_pred.extend(pred)
        offset=0
        for p,n in test_keys:
            for j in range(len(features[(p,n)])):
                prediction_rows.append(dict(fold=name,trial_filename=f'{p}/{p}_{n:02d}.txt',window_index=j,
                    true_label=p,predicted_label=str(pred[offset]),
                    error_ft=float(localization_errors([p],[pred[offset]])[0]),
                    weighted_x_ft=float(xy[offset,0]),weighted_y_ft=float(xy[offset,1]),
                    confidence=float(probabilities[offset].max())))
                offset+=1
        with (output/f'fold_{name}_confusion.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['true / predicted']+LABELS)
            writer.writerows([[p]+row.tolist() for p,row in zip(LABELS,cm)])
        ax=axes[index+1]; ax.imshow(cm,cmap='Blues')
        ax.set(xticks=range(9),yticks=range(9),xticklabels=LABELS,yticklabels=LABELS,
               xlabel='Predicted',ylabel='True',title=f'Fold {name}: {train:02d} → {test:02d}')
        for i in range(9):
            for j in range(9): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8)
    fig.savefig(output/'geometry_and_confusion.png',dpi=180);plt.close(fig)
    metric_keys=['exact_position_accuracy','mean_error_ft','median_error_ft','accuracy_within_2ft']
    means={k:float(np.mean([f[k] for f in folds])) for k in metric_keys}
    result=dict(folds=folds,fold_mean=means,
        mean_per_position_recall={p:float(np.mean([f['per_position_recall'][p] for f in folds])) for p in LABELS},
        confusion_matrix_labels=LABELS,
        mean_confusion_matrix=np.mean([f['confusion_matrix'] for f in folds],axis=0).tolist(),
        pooled_confusion_matrix=confusion_matrix(pooled_true,pooled_pred,labels=LABELS).tolist(),
        pooled_metrics=scores(pooled_true,pooled_pred))
    write_json(output/'metrics.json',result)
    with (output/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(prediction_rows[0]));writer.writeheader();writer.writerows(prediction_rows)
    lines=['# Diagonal geometry: trials 01 and 02','',
        '| Fold | Exact accuracy | Mean error (ft) | Median error (ft) | Within 2 ft |',
        '|---|---:|---:|---:|---:|']
    for name,row in [(f['fold'],f) for f in folds]+[('Mean',means)]:
        lines.append(f"| {name} | {row['exact_position_accuracy']:.2%} | {row['mean_error_ft']:.3f} | {row['median_error_ft']:.3f} | {row['accuracy_within_2ft']:.2%} |")
    lines+=['','Fold means are unweighted; mean median is the mean of the two fold medians. Pooled metrics are separate in metrics.json.',
        'Confusion matrices use P1–P9 row/column order. Rows are true labels.', '',
        '| Position | Fold A recall | Fold B recall | Mean recall |','|---|---:|---:|---:|']
    for p in LABELS:
        lines.append(f"| {p} | {folds[0]['per_position_recall'][p]:.2%} | {folds[1]['per_position_recall'][p]:.2%} | {result['mean_per_position_recall'][p]:.2%} |")
    lines+=['','Packet audit (each file):','',
        '| File | Lines | Non-CSI | Parsed | Corrupt | Wrong length | Accepted | Used | Trailing | Windows |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,a in audit.items():
        keys=['total_lines','non_csi_lines','parsed_packets','corrupt_packets','rejected_length_packets','accepted_packets','used_packets','trailing_packets','windows']
        lines.append('| '+name+' | '+' | '.join(str(a[k]) for k in keys)+' |')
    lines+=['','Complete corruption reasons and input hashes are in packet_audit.json.',
        'No model is loaded or saved. No recording or previous experiment is modified.',
        'Weighted XY values are raw probability interpolation; offline metrics use predicted class coordinates. Live confidence, EMA and UDP code is untouched.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print(f'\nResults: {output}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='/home/komail-shah/csi_localization_diagonal/')
    parser.add_argument('--output',default='/home/komail-shah/csi_localization_diagonal_evaluation/')
    args=parser.parse_args()
    run(args)


if __name__=='__main__':
    main()
