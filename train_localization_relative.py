#!/usr/bin/env python3
"""Fit the exact evaluator relative_shape pipeline on all original trials 01–05.

This final full-data fit is not an independent generalization evaluation.
Place beside evaluate_normalization.py. Never reads the old model.
"""
import argparse
import hashlib
import inspect
import json
import logging
import platform
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import evaluate_normalization as evaluation


def relative_contract():
    """Contract derived from the exact imported evaluator, shared with live code."""
    source = '\n'.join(inspect.getsource(getattr(evaluation, n)) for n in
                       ('variant_features', 'variant_names', 'extract_features',
                        'parse_csi_line', 'iq_to_amplitude', 'window_packets'))
    names = evaluation.variant_names('relative_shape')
    return dict(preprocessing_variant='relative_shape', preprocessing_version='relative_shape_v1',
                normalization=evaluation.FORMULAS['relative_shape'],
                normalization_parameters={'epsilon':evaluation.EPS, 'std_ddof':0, 'dtype':'float32'},
                preprocessing_implementation_sha256=hashlib.sha256(source.encode()).hexdigest(),
                raw_iq_length=2*evaluation.BINS, amplitude_length=evaluation.BINS,
                window_size_packets=evaluation.WINDOW, window_stride_packets=evaluation.WINDOW,
                feature_names=names, feature_count=len(names),
                feature_order=['mean_shape','relative_std','temporal_cv','relative_diff_abs_mean'],
                coordinates_ft=evaluation.COORDINATES,
                iq_format='alternating signed int8 components; hypot of each pair',
                std_ddof=0, quantile_method='linear', dtype='float32', subcarrier_mask='all',
                trailing_packets='discard',
                packet_policy='reject malformed/incomplete lines and length mismatches; concatenate accepted packets in original order')


def run(args):
    root=Path(args.data).expanduser().resolve()
    output=Path(args.output).expanduser().resolve()
    if root==output or root in output.parents or output in root.parents:
        raise ValueError('Output must be separate from the dataset.')
    output.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True,
                        handlers=[logging.StreamHandler(),logging.FileHandler(output/'training.log')])
    windows,audit=evaluation.load_data(root)
    evaluation.write_json(output/'packet_audit.json',audit)
    empty=[f'{p}/{p}_{n:02d}.txt' for (p,n),w in windows.items() if not len(w)]
    if empty: raise ValueError('No full window in: '+', '.join(empty))
    x=[]; y=[]; rows=[]
    for (p,n), ws in windows.items():
        for i,w in enumerate(ws):
            x.append(evaluation.variant_features(w,'relative_shape'))
            y.append(p);rows.append({'trial_filename':f'{p}/{p}_{n:02d}.txt','window_index':i})
    x=np.asarray(x,dtype=np.float32)
    metadata=relative_contract()
    if x.shape[1]!=metadata['feature_count'] or not np.isfinite(x).all():
        raise ValueError('Invalid relative_shape feature matrix')
    model=RandomForestClassifier(**evaluation.RF_PARAMETERS)
    model.fit(x,y)
    pred=model.predict(x)
    metadata.update(class_order=model.classes_.tolist(),class_labels=evaluation.LABELS,
                    n_features_in_=int(model.n_features_in_), model_parameters=model.get_params(),
                    train_trials=45,train_windows=len(y),training_trials=['01','02','03','04','05'],
                    numpy_version=np.__version__,sklearn_version=sklearn.__version__,
                    python_version=platform.python_version(),joblib_version=joblib.__version__,
                    validation='Full-data fit; no independent test. Use normalization cross-validation results for development estimates.')
    metrics=dict(training_accuracy=float(accuracy_score(y,pred)),train_trials=45,train_windows=len(y),
                 test_accuracy=None,warning='Training metrics only; no held-out evaluation in this final full-data fit.',
                 confusion_matrix_labels=evaluation.LABELS,
                 training_confusion_matrix=confusion_matrix(y,pred,labels=evaluation.LABELS).tolist(),
                 training_classification_report=classification_report(y,pred,labels=evaluation.LABELS,output_dict=True,zero_division=0))
    evaluation.write_json(output/'model_metadata.json',metadata)
    evaluation.write_json(output/'metrics.json',metrics)
    evaluation.write_json(output/'training_windows.json',rows)
    evaluation.write_json(output/'trial_split.json',{'train':[f'{p}/{p}_{n:02d}.txt' for p,n in windows],'test':[]})
    joblib.dump({'model':model,'metadata':metadata},output/'localization_model.joblib')
    logging.info('Trained relative_shape: %d trials, %d windows, %d features; training accuracy %.2f%%.',45,len(y),x.shape[1],100*metrics['training_accuracy'])
    logging.info('No independent test: this is the full-data final fit. Separate outputs: %s',output)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='/home/komail-shah/csi_localization/')
    parser.add_argument('--output',default='/home/komail-shah/csi_localization_relative_results/')
    args=parser.parse_args()
    try: run(args)
    except Exception:
        logging.exception('Relative-shape training failed; existing models/results were not overwritten.')
        raise SystemExit(1)


if __name__=='__main__': main()
