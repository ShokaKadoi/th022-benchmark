import csv
import json
from pathlib import Path
import tempfile
import unittest

import benchmark as b


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.labels = self.root / "labels.csv"
        self.submission = self.root / "submission.csv"
        self.targets = [{"participant_id": f"s{i}", "target": v, "split": "holdout"}
                        for i, v in enumerate([1, 2, 2, 4])]
        self.predictions = [{"participant_id": f"s{i}", "prediction": v, "half_a": v, "half_b": v}
                            for i, v in enumerate([1, 2, 2, 4])]

    def evaluate(self):
        b.write_table(self.labels, ["participant_id", "target", "split"], self.targets)
        b.write_table(self.submission, ["participant_id", "prediction", "half_a", "half_b"], self.predictions)
        return b.score(self.labels, self.submission)

    def test_perfect_tied_predictions(self):
        result = self.evaluate()
        self.assertAlmostEqual(result["prediction_spearman"], 1)
        self.assertFalse(result["rankable"])

    def test_order_independent(self):
        self.predictions.reverse()
        self.assertAlmostEqual(self.evaluate()["prediction_spearman"], 1)

    def test_reverse_predictions_preserve_sign(self):
        for r in self.predictions:
            r["prediction"] *= -1
        self.assertAlmostEqual(self.evaluate()["prediction_spearman"], -1)

    def test_constant_predictions_undefined(self):
        for r in self.predictions:
            r["prediction"] = 3
        self.assertIsNone(self.evaluate()["prediction_spearman"])

    def test_constant_half_undefined(self):
        for r in self.predictions:
            r["half_a"] = 1
        self.assertIsNone(self.evaluate()["split_half_pearson"])

    def test_nonfinite_rejected(self):
        for value in ["NaN", "inf", "-inf", ""]:
            with self.subTest(value=value):
                self.predictions[0]["prediction"] = value
                with self.assertRaises(ValueError):
                    self.evaluate()

    def test_duplicate_rejected(self):
        self.predictions.append(self.predictions[0])
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_missing_subject_rejected(self):
        self.predictions.pop()
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_train_subject_rejected(self):
        self.targets[0]["split"] = "train"
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_small_sample_rejected(self):
        self.targets = self.targets[:2]
        self.predictions = self.predictions[:2]
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_split_is_reproducible_disjoint_and_order_independent(self):
        ids = [f"s{i}" for i in range(103)]
        a = b.make_split(ids)
        self.assertEqual(a, b.make_split(list(reversed(ids))))
        self.assertEqual(sum(v == "holdout" for v in a.values()), 20)
        self.assertEqual(len(a), 103)

    def test_duplicate_split_id_rejected(self):
        with self.assertRaises(ValueError):
            b.make_split(["s1"] * 10)

    def test_tie_ranks(self):
        self.assertEqual(b.ranks([9, 1, 1, 4]), [4, 1.5, 1.5, 3])

    def test_large_finite_numbers(self):
        self.assertAlmostEqual(b.correlation([1e308, -1e308, 0], [1, -1, 0]), 1)

    def behavior(self, rows):
        path = self.root / "bds.tsv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["trial_index", "trialcode", "mean_span"])
            writer.writerows(rows)
        return path

    def test_terminal_span_not_column_average(self):
        self.assertEqual(b.bds_target(self.behavior([[3, "b_trial2", 0], [4, "b_exit", 6.25]])), 6.25)

    def test_incomplete_or_duplicate_exit_rejected(self):
        for rows in [[[3, "b_trial2", 0]], [[3, "b_exit", 6], [4, "b_exit", 6]],
                     [[3, "b_exit", 0]], [[3, "b_exit", 6], [4, "b_trial2", 0]]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                b.bds_target(self.behavior(rows))

    def test_trial_order_rejected(self):
        with self.assertRaises(ValueError):
            b.bds_target(self.behavior([[4, "b_trial2", 0], [3, "b_exit", 6]]))

    def test_duplicate_headers_rejected(self):
        path = self.root / "bad.csv"
        path.write_text("participant_id,target,target\na,1,2\n")
        with self.assertRaises(ValueError):
            b.read_table(path, ["participant_id", "target"])

    def test_prepare_excludes_missing_and_does_not_claim_eeg_download(self):
        dataset = self.root / "dataset"
        dataset.mkdir()
        (dataset / "dataset_description.json").write_text(json.dumps({
            "DatasetDOI": "doi:10.18112/openneuro.ds008037.v1.0.0", "License": "CC0"}))
        (dataset / "participants.tsv").write_text("participant_id\n" + "\n".join(f"sub-{i:03}" for i in range(8)) + "\n")
        for i in range(7):
            sid = f"sub-{i:03}"
            beh, eeg = dataset / sid / "beh", dataset / sid / "eeg"
            beh.mkdir(parents=True)
            eeg.mkdir()
            (beh / f"{sid}_task-bds_beh.tsv").write_text("trial_index\ttrialcode\tmean_span\n3\tb_exit\t6\n")
            (eeg / f"{sid}_task-tmseeg_acq-prefrontal_eeg.set").symlink_to("missing-annex-object")
            (eeg / f"{sid}_task-tmseeg_acq-prefrontal_events.tsv").write_text("onset\n1\n")
        out = self.root / "output"
        report = b.prepare(dataset, out)
        self.assertEqual(report["eligible_metadata_only"], 7)
        self.assertEqual(report["eeg_bytes_present"], 0)
        self.assertTrue((out / "organizer_only" / "labels.csv").exists())
        self.assertFalse((out / "labels.csv").exists())
        with self.assertRaises(FileExistsError):
            b.prepare(dataset, out)


if __name__ == "__main__":
    unittest.main()
