import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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

    def dataset_fixture(self):
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
        return dataset

    def test_prepare_excludes_missing_and_does_not_claim_eeg_download(self):
        dataset = self.dataset_fixture()
        out = self.root / "output"
        report = b.prepare(dataset, out, freeze_split=True)
        self.assertEqual(report["eligible_metadata_only"], 7)
        self.assertEqual(report["eeg_bytes_present"], 0)
        self.assertTrue((out / "organizer_only" / "labels.csv").exists())
        self.assertFalse((out / "labels.csv").exists())
        with self.assertRaises(FileExistsError):
            b.prepare(dataset, out, freeze_split=True)

    def frozen_fixture(self):
        dataset = self.dataset_fixture()
        first = self.root / "first"
        b.prepare(dataset, first, freeze_split=True)
        return dataset, first / "organizer_only" / "split_manifest.json"

    def test_rerun_preserves_split_without_recomputing(self):
        dataset, manifest = self.frozen_fixture()
        participants = dataset / "participants.tsv"
        lines = participants.read_text().splitlines()
        participants.write_text(lines[0] + "\n" + "\n".join(reversed(lines[1:])) + "\n")
        with patch.object(b, "make_split", side_effect=AssertionError("must not recalculate")):
            report = b.prepare(dataset, self.root / "rerun", split_manifest=manifest)
        self.assertEqual(report["split_mode"], "reuse")
        self.assertEqual(report["split_manifest_sha256"], b.digest(manifest))
        self.assertEqual((self.root / "first/organizer_only/labels.csv").read_bytes(),
                         (self.root / "rerun/organizer_only/labels.csv").read_bytes())

    def test_missing_train_or_holdout_subject_stops_before_output(self):
        dataset, manifest = self.frozen_fixture()
        frozen_bytes = manifest.read_bytes()
        splits = b.load_split(manifest)
        for split in ("train", "holdout"):
            sid = next(s for s in splits if splits[s] == split)
            path = dataset / sid / "beh" / f"{sid}_task-bds_beh.tsv"
            original = path.read_bytes()
            path.unlink()
            out = self.root / split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, sid):
                b.prepare(dataset, out, split_manifest=manifest)
            self.assertFalse(out.exists())
            self.assertEqual(manifest.read_bytes(), frozen_bytes)
            path.write_bytes(original)

    def test_removed_participant_stops(self):
        dataset, manifest = self.frozen_fixture()
        sid = next(iter(b.load_split(manifest)))
        participants = dataset / "participants.tsv"
        participants.write_text(participants.read_text().replace(sid + "\n", ""))
        with self.assertRaisesRegex(ValueError, "not_in_participants"):
            b.prepare(dataset, self.root / "removed", split_manifest=manifest)

    def test_invalid_behavior_stops(self):
        dataset, manifest = self.frozen_fixture()
        sid = next(iter(b.load_split(manifest)))
        (dataset / sid / "beh" / f"{sid}_task-bds_beh.tsv").write_text(
            "trial_index\ttrialcode\tmean_span\n3\tb_exit\tNaN\n")
        with self.assertRaisesRegex(ValueError, "bds_invalid"):
            b.prepare(dataset, self.root / "invalid", split_manifest=manifest)

    def test_added_eligible_subject_stops(self):
        import shutil
        dataset, manifest = self.frozen_fixture()
        shutil.copytree(dataset / "sub-000", dataset / "sub-007", symlinks=True)
        for path in list((dataset / "sub-007").rglob("*")):
            if path.name.startswith("sub-000"):
                path.rename(path.with_name(path.name.replace("sub-000", "sub-007")))
        with self.assertRaisesRegex(ValueError, "new_eligible.*sub-007"):
            b.prepare(dataset, self.root / "added", split_manifest=manifest)

    def test_split_mode_is_explicit_and_exclusive(self):
        with self.assertRaisesRegex(ValueError, "Choose exactly one"):
            b.prepare(self.root, self.root / "out")
        with self.assertRaisesRegex(ValueError, "Choose exactly one"):
            b.prepare(self.root, self.root / "out", freeze_split=True, split_manifest="missing.json")

    def test_invalid_split_manifest_rejected(self):
        dataset, manifest = self.frozen_fixture()
        original = json.loads(manifest.read_text())
        variants = [[], {**original, "protocol": "other"}, {**original, "source_snapshot": "other"},
                    {**original, "subjects": original["subjects"] + [original["subjects"][0]]},
                    {**original, "subjects": [{"participant_id": "sub-000", "split": "test"}]},
                    {**original, "subjects": original["subjects"][:2]}]
        for value in variants:
            manifest.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                b.prepare(dataset, self.root / "bad-manifest", split_manifest=manifest)
        self.assertFalse((self.root / "bad-manifest").exists())


if __name__ == "__main__":
    unittest.main()
