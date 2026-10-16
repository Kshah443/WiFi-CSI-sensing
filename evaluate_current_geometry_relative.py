#!/usr/bin/env python3
"""Two complete-trial folds on current geometry, using exact relative_shape code.

Place beside evaluate_normalization.py. No model is loaded or saved.
"""
import argparse
import csv
import hashlib
import inspect
import logging
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import evaluate_normalization as evaluation


def load_current(root):
    features, audits = {}, {}
    for position in evaluation.LABELS:
        for number in (6, 7):
            name = f'{position}/{position}_{number:02d}.txt'
            path = root / name
            packets, reasons, lengths, examples = [], Counter(), Counter(), []
            total = non_csi = parsed = rejected = 0
            with path.open(encoding='utf-8', errors='replace') as stream:
                for line_number, line in enumerate(stream, 1):
                    total += 1
                    try:
                        iq = evaluation.parse_csi_line(line)
                        if iq is None:
                            non_csi += 1
                            continue
                        amplitude = evaluation.iq_to_amplitude(iq)
                        parsed += 1
                        lengths[len(amplitude)] += 1
                        if len(amplitude) != evaluation.BINS:
                            rejected += 1
                            continue
                        packets.append(amplitude)
                    except (ValueError, OverflowError) as exc:
                        reasons[str(exc)] += 1
                        if len(examples) < 10:
                            examples.append({'line': line_number, 'reason': str(exc)})
            windows = evaluation.window_packets(
                np.asarray(packets, dtype=np.float32).reshape(-1, evaluation.BINS), evaluation.WINDOW)
            audits[name] = dict(total_lines=total, non_csi_lines=non_csi, parsed_packets=parsed,
                                corrupt_packets=sum(reasons.values()), corruption_reasons=dict(reasons),
                                corruption_examples=examples, amplitude_length_histogram=dict(lengths),
                                rejected_length_packets=rejected, accepted_packets=len(packets),
                                used_packets=len(windows)*evaluation.WINDOW,
                                trailing_packets=len(packets)%evaluation.WINDOW, windows=len(windows),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            features[(position, number)] = np.asarray(
                [evaluation.variant_features(w, 'relative_shape') for w in windows], dtype=np.float32)
    return features, audits


def assemble(features, number):
    x = np.concatenate([features[(p, number)] for p in evaluation.LABELS])
    y = np.concatenate([np.repeat(p, len(features[(p, number)])) for p in evaluation.LABELS])
    rows = [{'trial_filename': f'{p}/{p}_{number:02d}.txt', 'window_index': i}
            for p in evaluation.LABELS for i in range(len(features[(p, number)]))]
    return x, y, rows


def run(args):
    root, output = Path(args.data).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Output must be separate from the dataset.')
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(output/'evaluation.log')])
    if evaluation.BINS != 117 or evaluation.WINDOW != 100:
        raise ValueError('Evaluator must specify 117 amplitude bins and 100-packet windows.')
    features, audit = load_current(root)
    evaluation.write_json(output/'packet_audit.json', audit)
    empty = [name for name, row in audit.items() if not row['windows']]
    if empty:
        raise ValueError('No complete 117-bin window in: ' + ', '.join(empty))
    names = evaluation.variant_names('relative_shape')
    if any(x.shape[1] != len(names) or not np.isfinite(x).all() for x in features.values()):
        raise ValueError('Invalid relative_shape feature matrix.')
    implementation = '\n'.join(inspect.getsource(getattr(evaluation, n)) for n in
                               ('variant_features', 'variant_names', 'parse_csi_line', 'iq_to_amplitude', 'window_packets'))
    evaluation.write_json(output/'experiment_manifest.json', dict(
        preprocessing_variant='relative_shape', normalization_formula=evaluation.FORMULAS['relative_shape'],
        normalization_parameters={'epsilon': evaluation.EPS, 'std_ddof': 0, 'dtype': 'float32'},
        raw_iq_length=234, amplitude_length=117, window_size_packets=100, window_stride_packets=100,
        feature_names=names, feature_count=len(names), coordinates_ft=evaluation.COORDINATES,
        random_forest_parameters=evaluation.RF_PARAMETERS,
        implementation_sha256=hashlib.sha256(implementation.encode()).hexdigest(),
        validation='A: train 06/test 07; B: train 07/test 06. Entire trial numbers held out at every position.',
        mean_policy='Arithmetic mean across two folds; mean confusion matrix is elementwise mean of count matrices. Pooled matrix is separately labeled.'))
    folds, matrices = [], []
    with (output/'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['fold', 'trial_filename', 'window_index', 'true_label', 'predicted_label', 'error_ft'])
        writer.writeheader()
        for label, train_number, test_number in [('A', 6, 7), ('B', 7, 6)]:
            train_x, train_y, _ = assemble(features, train_number)
            test_x, test_y, rows = assemble(features, test_number)
            model = RandomForestClassifier(**evaluation.RF_PARAMETERS)
            model.fit(train_x, train_y)
            predictions = model.predict(test_x)
            errors = evaluation.localization_errors(test_y, predictions)
            cm = confusion_matrix(test_y, predictions, labels=evaluation.LABELS)
            report = classification_report(test_y, predictions, labels=evaluation.LABELS, output_dict=True, zero_division=0)
            fold = dict(fold=label, train_trial=f'{train_number:02d}', test_trial=f'{test_number:02d}',
                        train_trials=9, test_trials=9, train_windows=len(train_y), test_windows=len(test_y),
                        training_accuracy=float(accuracy_score(train_y, model.predict(train_x))),
                        **evaluation.scores(test_y, predictions), confusion_matrix=cm.tolist(),
                        per_position_recall={p:report[p]['recall'] for p in evaluation.LABELS},
                        classification_report=report, class_order=model.classes_.tolist())
            folds.append(fold); matrices.append(cm)
            for row, truth, prediction, error in zip(rows, test_y, predictions, errors):
                writer.writerow(dict(row, fold=label, true_label=truth, predicted_label=prediction, error_ft=float(error)))
            logging.info('Fold %s: train %02d/test %02d; train %.2f%%; exact %.2f%%; mean %.3f ft; median %.3f ft; within 2 ft %.2f%%',
                         label, train_number, test_number, fold['training_accuracy']*100, fold['exact_position_accuracy']*100,
                         fold['mean_error_ft'], fold['median_error_ft'], fold['accuracy_within_2ft']*100)
    keys = ['training_accuracy', 'exact_position_accuracy', 'mean_error_ft', 'median_error_ft', 'accuracy_within_2ft']
    means = {k:float(np.mean([f[k] for f in folds])) for k in keys}
    means['confusion_matrix'] = np.mean(matrices, axis=0).tolist()
    means['per_position_recall'] = {p:float(np.mean([f['per_position_recall'][p] for f in folds])) for p in evaluation.LABELS}
    pooled = np.sum(matrices, axis=0)
    result = dict(folds=folds, arithmetic_fold_mean=means, confusion_matrix_labels=evaluation.LABELS,
                  aggregated_confusion_matrix=pooled.tolist(),
                  pooled_per_position_recall={p:float(pooled[i,i]/pooled[i].sum()) for i,p in enumerate(evaluation.LABELS)},
                  limitation='Small two-fold geometry experiment: one training and one held-out trial per position per fold. Adjacent windows are correlated; folds reuse the same recordings in reversed roles. Compare directionally only; no independent confirmation of selecting relative_shape.')
    evaluation.write_json(output/'metrics.json', result)
    evaluation.write_json(output/'trial_splits.json', [dict(fold=f['fold'], train=[f'{p}/{p}_{f["train_trial"]}.txt' for p in evaluation.LABELS],
                                                         test=[f'{p}/{p}_{f["test_trial"]}.txt' for p in evaluation.LABELS]) for f in folds])
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['fold']+keys); writer.writeheader()
        for f in folds: writer.writerow({'fold':f['fold'], **{k:f[k] for k in keys}})
        writer.writerow({'fold':'mean', **{k:means[k] for k in keys}})
    lines=['# Current geometry: relative-shape evaluation', '', result['limitation'], '',
           '| Fold | Train accuracy | Exact accuracy | Mean error ft | Median error ft | Within 2 ft |',
           '|---|---:|---:|---:|---:|---:|']
    for name,row in [(f['fold'],f) for f in folds]+[('Mean',means)]:
        lines.append(f"| {name} | {row['training_accuracy']:.2%} | {row['exact_position_accuracy']:.2%} | {row['mean_error_ft']:.3f} | {row['median_error_ft']:.3f} | {row['accuracy_within_2ft']:.2%} |")
    lines += ['', 'Mean is arithmetic across folds, including the mean of the two fold medians. Accuracy values in JSON/CSV are fractions.', '',
              '| Fold | '+' | '.join(evaluation.LABELS)+' |', '|---|'+'---:|'*9]
    for name,row in [(f['fold'],f) for f in folds]+[('Mean',means)]:
        lines.append('| '+name+' | '+' | '.join(f"{row['per_position_recall'][p]:.2%}" for p in evaluation.LABELS)+' |')
    for name,matrix in [(f['fold'],f['confusion_matrix']) for f in folds]+[('Mean',means['confusion_matrix'])]:
        lines += ['', f'## {name} confusion matrix', '', 'Rows are true positions; columns are predictions. Mean cells may be fractional.', '',
                  '| True / predicted | '+' | '.join(evaluation.LABELS)+' |', '|---|'+'---:|'*9]
        for p,row in zip(evaluation.LABELS,matrix): lines.append('| '+p+' | '+' | '.join(f'{v:g}' for v in row)+' |')
        with (output/f'confusion_matrix_{name.lower()}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['true / predicted']+evaluation.LABELS)
            for p,row in zip(evaluation.LABELS,matrix):writer.writerow([p]+list(row))
    lines += ['', 'Exact preprocessing is imported from evaluate_normalization.py. No model artifact is loaded, saved, or modified.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    logging.info('Results saved to %s',output)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='/home/komail-shah/csi_localization/')
    parser.add_argument('--output',default='/home/komail-shah/csi_localization_current_geometry_relative_test/')
    args=parser.parse_args()
    try:run(args)
    except Exception:
        logging.exception('Geometry evaluation failed; existing outputs were not overwritten.')
        raise SystemExit(1)


if __name__=='__main__':main()
