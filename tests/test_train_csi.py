"""Regression checks for corrupt inputs, window boundaries, and trial leakage."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

import train_csi as csi


class CsiTrainingTests(unittest.TestCase):
    def test_amplitude_accepts_prefix_and_ansi_and_uses_float_arithmetic(self):
        actual = csi.parse_packet('\x1b[32mlog: CSI_DATA,1,"[3,4,-32768,0]"\x1b[0m\n')
        np.testing.assert_array_equal(actual, [5.0, 32768.0])

    def test_rejects_corrupted_records(self):
        records = [
            'CSI_DATA,1,"[1,2,3]"', 'CSI_DATA,1,"[]"',
            'CSI_DATA,1,"[1,2,bad,4]"', 'CSI_DATA,1,"[1,2]"junk',
            'CSI_DATA,1,"[1,2]', 'CSI_DATA,1,"[1,\ufffd2]"',
            'CSI_DATA,1,"[1,2]"\x00', 'CSI_DATA,1,"[1,99999]"',
            'CSI_DATA,1,"[1,2]" CSI_DATA,2,"[3,4]"',
            'CSI_DATA,1,"[[1,2]]"', 'CSI_DATA,1,"[True,2]"',
        ]
        for record in records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                csi.parse_packet(record)

    def test_statistics_and_difference_order(self):
        features = csi.extract_features(np.array([[1., 5.], [3., 1.]]))
        np.testing.assert_allclose(features, [
            2, 3, 1, 2, 1, 4, 1, 1, 3, 5, 2, 3,
            2, -4, 0, 0, 2, 4, 2, 4,
        ])
        self.assertEqual(csi.extract_features(np.ones((100, 3)), False).shape, (18,))

    def test_filtering_and_windows_do_not_cross_files(self):
        trials = []
        for index, count in enumerate((205, 95)):
            trial = csi.Trial(Path(f"trial_{index}.txt"), "empty")
            trial.packets = [np.array([float(i), float(i)]) for i in range(count)]
            trial.packets.insert(30, np.ones(3))
            trial.audit = {"skip_reason": None}
            trials.append(trial)
        usable = csi.build_windows(trials, 4, 100, False)
        self.assertEqual(len(usable), 1)
        self.assertEqual(usable[0].features.shape, (2, 12))
        np.testing.assert_allclose(usable[0].features[:, 0], [49.5, 149.5])
        self.assertEqual(trials[0].audit["trailing_packets_discarded"], 5)
        self.assertEqual(trials[1].audit["trailing_packets_discarded"], 95)
        self.assertEqual(trials[0].audit["length_mismatch_packets"], 1)

    def test_split_is_reproducible_disjoint_and_has_all_classes(self):
        trials = [csi.Trial(Path(f"/{label}/{i}.txt"), label)
                  for label in csi.LABELS for i in range(5)]
        train, test = csi.split_trials(trials, 0.2, 42)
        train_again, test_again = csi.split_trials(trials, 0.2, 42)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(test), 3)
        self.assertFalse({t.path for t in train} & {t.path for t in test})
        self.assertEqual({t.label for t in test}, set(csi.LABELS))
        self.assertEqual([t.path for t in train], [t.path for t in train_again])
        self.assertEqual([t.path for t in test], [t.path for t in test_again])
        with self.assertRaisesRegex(ValueError, "at least two"):
            csi.split_trials(trials[:1], 0.2, 42)

    def test_packet_audit_and_unreadable_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.txt"
            path.write_bytes(b'boot log\n\xff\nCSI_DATA,1,"[3,4]"\nCSI_DATA,2,"[1,2,3]"\n')
            trial = csi.read_trial(path, "empty")
            self.assertEqual(trial.audit["non_csi_lines"], 2)
            self.assertEqual(trial.audit["damaged_non_csi_lines"], 1)
            self.assertEqual(trial.audit["valid_packets"], 1)
            self.assertEqual(trial.audit["corrupt_packets"], 1)
            missing = csi.read_trial(Path(directory) / "missing.txt", "empty")
            self.assertIsNotNone(missing.audit["file_error"])
            self.assertFalse(missing.packets)

    def test_end_to_end_artifacts_and_saved_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, label in enumerate(csi.LABELS):
                folder = root / label
                folder.mkdir()
                for trial in range(2):
                    records = [f'CSI_DATA,{i},"[{index * 30 + i % 3},4,3,4]"\n'
                               for i in range(205)]
                    records += ['CSI_DATA,206,"[3,4]"\n', 'CSI_DATA,207,"[broken]"\n']
                    (folder / f"{trial}.txt").write_text("".join(records), encoding="utf-8")
            output = root / "results"
            with contextlib.redirect_stdout(io.StringIO()):
                status = csi.main(["--dataset", str(root), "--output", str(output),
                                   "--n-estimators", "5", "--n-jobs", "1"])
            self.assertEqual(status, 0)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["train_windows"], 6)
            self.assertEqual(metrics["test_windows"], 6)
            self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 6)
            audit = json.loads((output / "packet_audit.json").read_text())
            self.assertEqual(audit["selected_iq_length"], 4)
            self.assertEqual(audit["totals"]["corrupt_packets"], 6)
            self.assertEqual(audit["totals"]["length_mismatch_packets"], 6)
            self.assertTrue((output / "confusion_matrix.png").read_bytes().startswith(b"\x89PNG"))
            model = joblib.load(output / "random_forest.joblib")
            self.assertEqual(model.n_features_in_, 20)
            prediction = model.predict([csi.extract_features(np.ones((100, 2)))])
            self.assertIn(prediction[0], csi.LABELS)
            split = json.loads((output / "trial_split.json").read_text())
            self.assertFalse({t["path"] for t in split["train"]} & {t["path"] for t in split["test"]})


if __name__ == "__main__":
    unittest.main()
