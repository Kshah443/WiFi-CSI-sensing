#!/usr/bin/env python3
"""Compare normalization with complete-trial-held-out CSI validation.
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
COORDINATES = {label: (-2 + 2 * (i % 3), 2 + 2 * (i // 3)) for i, label in enumerate(LABELS)}
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
        for number in range(1, 6):
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


def drift_diagnostic(features, held_out):
    # Train-only coordinate standardization makes unlike feature units comparable.
    train = np.concatenate([x for (p, n), x in features.items() if n != held_out])
    center, scale = train.mean(0), train.std(0)
    active = scale > EPS
    if not active.any():
        return dict(active_features=0, same_position_distance=0., different_position_distance=0.,
                    same_to_different_ratio=None, note='No variable training features.')
    centroids = {key: ((x.mean(0)-center)[active]/scale[active]) for key, x in features.items()}
    same, different = [], []
    for p in LABELS:
        for n in range(1, 6):
            if n == held_out:
                continue
            same.append(float(np.linalg.norm(centroids[(p,held_out)]-centroids[(p,n)])/np.sqrt(active.sum())))
            for q in LABELS:
                if q != p:
                    different.append(float(np.linalg.norm(centroids[(p,held_out)]-centroids[(q,n)])/np.sqrt(active.sum())))
    s, d = float(np.mean(same)), float(np.mean(different))
    return dict(active_features=int(active.sum()), same_position_distance=s,
                different_position_distance=d, same_to_different_ratio=s/d if d > EPS else None)


def run(args):
    root, output = Path(args.data).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Use an output directory separate from the dataset.')
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(output/'evaluation.log')])
    logging.info('Loading only trials 01–05; fixed 117 bins, 100-packet nonoverlapping windows.')
    windows, audit = load_data(root)
    write_json(output/'packet_audit.json', audit)
    empty = [f'{p}/{p}_{n:02d}.txt' for (p,n), w in windows.items() if not len(w)]
    if empty:
        raise ValueError('No complete 117-bin window in: '+', '.join(empty))
    splits = [dict(fold=n, train=[f'{p}/{p}_{i:02d}.txt' for p in LABELS for i in range(1,6) if i != n],
                   test=[f'{p}/{p}_{n:02d}.txt' for p in LABELS]) for n in range(1,6)]
    write_json(output/'trial_splits.json', splits)
    manifest = dict(formulas=FORMULAS, epsilon=EPS, amplitude_bins=BINS, raw_iq_length=234,
                    window_size=WINDOW, window_stride=WINDOW, trailing_packets='discard per file',
                    preprocessing_dtype='float32', std_ddof=0, quantile_method='linear',
                    packet_policy='Original parser; reject malformed/incomplete and non-117-bin packets; retain accepted packet order.',
                    coordinates_ft=COORDINATES, random_forest_parameters=RF_PARAMETERS,
                    feature_names={v:variant_names(v) for v in FORMULAS},
                    feature_counts={v:len(variant_names(v)) for v in FORMULAS},
                    validation='Five exhaustive leave-one-trial-number-out folds: hold out the same complete number at all nine positions. No random window splits.',
                    drift_formula='Trial centroids in feature space; train-only feature mean/std standardization; omit features with training std <= epsilon. RMS standardized distance from each test trial centroid to same-position vs different-position training trial centroids. Ratio near or above 1 indicates weak position separation relative to cross-trial drift.',
                    versions=dict(numpy=np.__version__, sklearn=sklearn.__version__, python=platform.python_version()),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output/'experiment_manifest.json', manifest)
    all_results, summary_rows = {}, []
    with (output/'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant','fold','trial_filename','window_index','true_label','predicted_label','error_ft'])
        writer.writeheader()
        for variant in FORMULAS:
            features = {key:np.asarray([variant_features(w, variant) for w in ws]) for key,ws in windows.items()}
            if not all(np.isfinite(x).all() for x in features.values()):
                raise ValueError(f'Nonfinite features: {variant}')
            folds, truths, predictions = [], [], []
            aggregate_cm = np.zeros((9,9), dtype=int)
            for n in range(1,6):
                train_keys = [key for key in features if key[1] != n]
                test_keys = [key for key in features if key[1] == n]
                x_train = np.concatenate([features[k] for k in train_keys])
                y_train = np.concatenate([np.repeat(k[0], len(features[k])) for k in train_keys])
                x_test = np.concatenate([features[k] for k in test_keys])
                y_test = np.concatenate([np.repeat(k[0], len(features[k])) for k in test_keys])
                model = RandomForestClassifier(**RF_PARAMETERS)
                model.fit(x_train, y_train)
                train_pred, pred = model.predict(x_train), model.predict(x_test)
                cm = confusion_matrix(y_test, pred, labels=LABELS)
                aggregate_cm += cm
                fold = dict(fold=n, held_out_trial=f'{n:02d}', train_windows=len(y_train), test_windows=len(y_test),
                            train_accuracy=float(accuracy_score(y_train, train_pred)), **scores(y_test,pred),
                            confusion_matrix=cm.tolist(), classification_report=classification_report(y_test,pred,labels=LABELS,output_dict=True,zero_division=0),
                            feature_drift=drift_diagnostic(features,n))
                folds.append(fold); truths.extend(y_test); predictions.extend(pred)
                offset = 0
                for p,i in test_keys:
                    for j in range(len(features[(p,i)])):
                        predicted = str(pred[offset])
                        writer.writerow(dict(variant=variant,fold=n,trial_filename=f'{p}/{p}_{i:02d}.txt',window_index=j,
                                             true_label=p,predicted_label=predicted,error_ft=float(localization_errors([p],[predicted])[0])))
                        offset += 1
                logging.info('%s fold %d: train %.2f%% test %.2f%% mean error %.3f ft', variant,n,fold['train_accuracy']*100,fold['exact_position_accuracy']*100,fold['mean_error_ft'])
            keys = ['exact_position_accuracy','train_accuracy','mean_error_ft','median_error_ft','accuracy_within_2ft']
            # Fold-balanced summaries; pooled metrics are also supplied explicitly.
            summary = {k+'_mean':float(np.mean([f[k] for f in folds])) for k in keys}
            summary.update({k+'_std':float(np.std([f[k] for f in folds],ddof=0)) for k in keys})
            report = classification_report(truths,predictions,labels=LABELS,output_dict=True,zero_division=0)
            ratios = [f['feature_drift']['same_to_different_ratio'] for f in folds if f['feature_drift']['same_to_different_ratio'] is not None]
            summary['feature_drift_ratio_mean'] = float(np.mean(ratios)) if ratios else None
            result = dict(formula=FORMULAS[variant],folds=folds,fold_summary=summary,
                          pooled_test_metrics=scores(truths,predictions), confusion_matrix_labels=LABELS,
                          aggregated_confusion_matrix=aggregate_cm.tolist(), pooled_classification_report=report,
                          per_position_recall={p:report[p]['recall'] for p in LABELS})
            all_results[variant] = result
            write_json(output/f'{variant}_metrics.json', result)
            with (output/f'{variant}_confusion_matrix.csv').open('w',newline='') as cm_file:
                cm_writer=csv.writer(cm_file); cm_writer.writerow(['true / predicted']+LABELS)
                for p,row in zip(LABELS,aggregate_cm): cm_writer.writerow([p]+row.tolist())
            summary_rows.append(dict(variant=variant,**summary))
    write_json(output/'metrics.json', all_results)
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
    lines = ['# CSI normalization experiment', '',
             'Five exhaustive trial-held-out folds on original trials 01–05. Each fold tests one complete trial number at every position and trains on the other four. Trials 06–07 are excluded.', '',
             'Fold means are unweighted across five folds; standard deviations use ddof=0. Accuracy values in JSON/CSV are fractions. Pooled metrics combine all held-out predictions; pooled median differs from the mean of fold medians.', '',
             '| Variant | Train mean | Test mean ± SD | Mean error ft | Mean fold median ft | Within 2 ft | Drift ratio |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary_rows:
        ratio=row['feature_drift_ratio_mean']; ratio_text=f'{ratio:.3f}' if ratio is not None else 'undefined'
        lines.append(f"| {row['variant']} | {row['train_accuracy_mean']:.2%} | {row['exact_position_accuracy_mean']:.2%} ± {100*row['exact_position_accuracy_std']:.2f} pp | {row['mean_error_ft_mean']:.3f} | {row['median_error_ft_mean']:.3f} | {row['accuracy_within_2ft_mean']:.2%} | {ratio_text} |")
    lines.extend(['','## Per-position recall (pooled held-out predictions)','', '| Variant | '+' | '.join(LABELS)+' |', '|---|'+'---:|'*9])
    for v,r in all_results.items(): lines.append('| '+v+' | '+' | '.join(f"{r['per_position_recall'][p]:.2%}" for p in LABELS)+' |')
    lines.extend(['','## Formulas','', 'A is nonnegative amplitude, p indexes packets and b indexes 117 bins. epsilon = 1e-8. Standard deviation uses ddof=0. Each window has 100 packets.'])
    for v,formula in FORMULAS.items(): lines.extend(['',f'- **{v}**: {formula}'])
    lines.extend(['','The first five variants use the original feature statistics and order unchanged. Relative shape uses 468 features in the order recorded in experiment_manifest.json; all other variants use 1,290 features.', '',
                  'Drift is computed using training-only feature scaling in each fold. Compare held-out trial centroids against same-position and different-position training centroids. Smaller same/different ratios suggest more stable position structure. This diagnostic is sensitive to the feature representation and cannot prove gain is the cause.', '',
                  'An improvement in normalized variants would support amplitude scale as a contributor to cross-trial failure. A lack of improvement does not rule it out: multipath changes, orientation, timing, and within-trial variation can also affect performance. Normalization may remove useful spatial signal.', '',
                  'These five folds share training trials and adjacent windows are correlated. Fold SD is descriptive, not a confidence interval. Choosing the best variant on these results makes these folds development data; confirm it on new untouched trials before claiming final performance. These folds do not establish generalization to new rooms or antenna placements.', '',
                  'No model artifact is loaded or saved. Inputs and existing results are untouched.'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:15]), flush=True)
    logging.info('Finished. Results: %s', output)
    return all_results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='/home/komail-shah/csi_localization/')
    parser.add_argument('--output',default='/home/komail-shah/csi_localization_normalization_test/')
    args=parser.parse_args()
    try: run(args)
    except Exception:
        logging.exception('Evaluation failed; no existing results were overwritten.')
        raise SystemExit(1)


if __name__ == '__main__':
    main()

