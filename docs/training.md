# CSI training

From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-csi.txt
.venv/bin/python train_csi.py
```

The defaults read every `.txt` trial directly inside
`/home/komail-shah/csi_dataset/{empty,still,walking}` and write to
`/home/komail-shah/csi_ml_results/`. Re-running replaces the named output files.

The parser accepts complete `CSI_DATA` records ending in a quoted, flat list of
signed decimal integers, strips terminal colors, and rejects damaged text,
incomplete I/Q pairs, and samples outside the firmware's signed 16-bit range.
It computes amplitude with floating-point `hypot` to avoid integer overflow.
The most frequent I/Q vector length across readable trials is retained; ties
choose the smaller length.

Each trial is divided into non-overlapping windows of 100 retained packets.
Rejected packets are removed first, so a window can span missing packets in the
original capture. Windows never cross trial boundaries. Incomplete final
windows are counted and discarded. Each subcarrier contributes mean, population
standard deviation, population variance, minimum, maximum, and median, plus
signed difference mean/std and absolute difference mean/max. Temporal
differences are calculated only within each window and are per packet, not per
second. Use `--no-temporal` to disable them.

The split shuffles **whole trial files within each class**, using seed 42.
Twenty percent of each class's usable trials, rounded up with at least one
training trial remaining, become the test set. With five usable trials per
class this means four training trials and one test trial per class. At least
two usable trials per class are required; the script fails clearly otherwise.
It verifies that the file sets are disjoint. Only training features are used
to fit the random forest. The dataset-wide modal length is a format filter.

```bash
.venv/bin/python train_csi.py --help
.venv/bin/python train_csi.py --test-size 0.4 --seed 7 --no-temporal
.venv/bin/python -m unittest -v test_train_csi.py
```

Outputs:

- `random_forest.joblib`: fitted `RandomForestClassifier`, trained on the training split.
- `confusion_matrix.png`: held-out window counts; rows are true labels.
- `metrics.json`: train/test trial and window counts, accuracy, report, and matrix.
- `model_metadata.json`: vector length, window size, exact feature ordering, and versions.
- `trial_split.json`: paths, labels, and window counts for each partition.
- `packet_audit.json`: per-file parsing errors with example line numbers, vector lengths,
  skipped packets/files, incomplete window counts, and totals.
- `training.log`: the printed diagnostics and evaluation.

To predict a new window, use the same parsing and modal length, collect exactly
the saved window size, and apply the same feature options:

```python
import json
from pathlib import Path
import joblib
import numpy as np
from train_csi import extract_features, parse_packet

results = Path('/home/komail-shah/csi_ml_results')
metadata = json.loads((results / 'model_metadata.json').read_text())
model = joblib.load(results / 'random_forest.joblib')
# packet_lines contains one window of complete, valid CSI records.
amplitudes = np.stack([parse_packet(line) for line in packet_lines])
assert amplitudes.shape == (metadata['window_size'], metadata['subcarriers'])
features = extract_features(amplitudes, temporal=metadata['temporal_features'])
label = model.predict(features.reshape(1, -1))[0]
```

Accuracy measures windows from held-out recordings of this dataset. Trials
from other rooms, sessions, device placements, or people are useful additional
tests of generalization. The script preserves subcarrier array order and does
not infer or remove firmware-specific guard/null subcarriers.
