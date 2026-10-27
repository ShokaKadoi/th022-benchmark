"""Exercise submission -> scoring with clearly synthetic fixtures, never real EEG."""
import json
from pathlib import Path
import tempfile

import benchmark as b


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        b.write_table(root / "labels.csv", ["participant_id", "target", "split"],
                      [{"participant_id": f"synthetic-{i}", "target": i, "split": "holdout"} for i in range(8)])
        b.write_table(root / "submission.csv", ["participant_id", "prediction", "half_a", "half_b"],
                      [{"participant_id": f"synthetic-{i}", "prediction": p, "half_a": i, "half_b": i + (-1)**i * 0.2}
                       for i, p in enumerate([0, 2, 1, 3, 5, 4, 6, 7])])
        result = b.score(root / "labels.csv", root / "submission.csv")
        result["data_kind"] = "SYNTHETIC_PIPELINE_TEST_NOT_SCIENTIFIC_EVIDENCE"
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
