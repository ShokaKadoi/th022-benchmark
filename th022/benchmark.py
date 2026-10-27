"""TH-022 v0.1: offline preparation and evaluation, Python 3.10+ stdlib only."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

VERSION = "th022-bds-prefrontal-v0.1"
SNAPSHOT = "98931f03b602a7685d63f88a85b71dfd35872a11"


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite number")
    return value


def read_table(path, columns, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate headers: {path}")
        if not set(columns).issubset(reader.fieldnames):
            raise ValueError(f"Missing columns {columns}: {path}")
        rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Empty or malformed table: {path}")
    return rows


def write_table(path, columns, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bds_target(path):
    rows = read_table(path, ["trial_index", "trialcode", "mean_span"], "\t")
    indices = [number(row["trial_index"]) for row in rows]
    if any(not x.is_integer() for x in indices) or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("Trial indices must strictly increase")
    exits = [row for row in rows if row["trialcode"] == "b_exit"]
    if len(exits) != 1 or rows[-1] != exits[0]:
        raise ValueError("Require exactly one terminal b_exit row")
    target = number(exits[0]["mean_span"])
    if target <= 0:
        raise ValueError("Terminal mean_span must be positive")
    return target


def make_split(ids, seed="th022-v0.1", fraction=0.2):
    if len(set(ids)) != len(ids) or len(ids) < 6 or not 0 < fraction < 1:
        raise ValueError("Need unique IDs, >=6 subjects, and 0<fraction<1")
    ordered = sorted(ids, key=lambda sid: (hashlib.sha256((seed + ":" + sid).encode()).hexdigest(), sid))
    count = max(3, int(len(ids) * fraction))
    if len(ids) - count < 3:
        raise ValueError("Both splits need at least three subjects")
    test = set(ordered[:count])
    return {sid: ("holdout" if sid in test else "train") for sid in sorted(ids)}


def prepare(root, output):
    root, output = Path(root), Path(output)
    metadata = json.loads((root / "dataset_description.json").read_text())
    if metadata.get("DatasetDOI") != "doi:10.18112/openneuro.ds008037.v1.0.0":
        raise ValueError("Expected ds008037 version 1.0.0")
    participants = read_table(root / "participants.tsv", ["participant_id"], "\t")
    ids = [r["participant_id"] for r in participants]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate participant IDs")
    labels, exclusions, manifests = {}, [], []
    for sid in sorted(ids):
        if not sid.startswith("sub-") or not sid[4:].isdigit():
            raise ValueError("Invalid BIDS participant ID")
        behavior = root / sid / "beh" / f"{sid}_task-bds_beh.tsv"
        eeg = root / sid / "eeg" / f"{sid}_task-tmseeg_acq-prefrontal_eeg.set"
        events = eeg.with_name(eeg.name.replace("_eeg.set", "_events.tsv"))
        reasons = []
        try:
            target = bds_target(behavior)
        except (OSError, ValueError) as exc:
            reasons.append("bds_missing" if not behavior.exists() else "bds_invalid: " + str(exc))
        if not (eeg.exists() or eeg.is_symlink()):
            reasons.append("prefrontal_eeg_not_listed")
        if not events.exists():
            reasons.append("prefrontal_events_missing")
        if reasons:
            exclusions.append({"participant_id": sid, "reasons": reasons})
            continue
        labels[sid] = target
        manifests.append({"participant_id": sid, "behavior_path": str(behavior.relative_to(root)),
                          "behavior_sha256": digest(behavior), "eeg_path": str(eeg.relative_to(root)),
                          "eeg_bytes_present": eeg.exists(), "events_sha256": digest(events)})
    splits = make_split(list(labels))
    # Never overwrite a previous preparation or mix versions in a directory.
    output.mkdir(parents=True, exist_ok=False)
    private = output / "organizer_only"
    private.mkdir()
    write_table(output / "train_labels.csv", ["participant_id", "target"],
                [{"participant_id": sid, "target": labels[sid]} for sid in splits if splits[sid] == "train"])
    write_table(output / "holdout_ids.csv", ["participant_id"],
                [{"participant_id": sid} for sid in splits if splits[sid] == "holdout"])
    write_table(private / "labels.csv", ["participant_id", "target", "split"],
                [{"participant_id": sid, "target": labels[sid], "split": splits[sid]} for sid in splits])
    report = {"protocol": VERSION, "expected_source_commit": SNAPSHOT,
              "source_commit_verified_by_tool": False,
              "dataset_license_metadata": metadata.get("License"),
              "participants": len(ids), "eligible_metadata_only": len(labels),
              "train": sum(s == "train" for s in splits.values()),
              "holdout": sum(s == "holdout" for s in splits.values()),
              "eeg_bytes_present": sum(m["eeg_bytes_present"] for m in manifests),
              "status": "METADATA_ONLY_NOT_READY_FOR_COMPETITION",
              "target": "Terminal b_exit mean_span; provisional BDS endpoint, not general WM capacity",
              "exclusions": exclusions, "manifest": manifests}
    (private / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k not in ("manifest", "exclusions")}


def ranks(values):
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for i in ordered[start:end]:
            result[i] = (start + end - 1) / 2 + 1
        start = end
    return result


def correlation(a, b):
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("Need >=3 matched observations")
    a, b = [number(x) for x in a], [number(x) for x in b]
    # Normalize before centering to avoid overflow for finite extreme inputs.
    sa, sb = max(map(abs, a)), max(map(abs, b))
    if sa == 0 or sb == 0:
        return None
    a, b = [x / sa for x in a], [x / sb for x in b]
    ma, mb = statistics.mean(a), statistics.mean(b)
    a, b = [x - ma for x in a], [x - mb for x in b]
    denom = math.sqrt(math.fsum(x*x for x in a) * math.fsum(x*x for x in b))
    if denom == 0:
        return None
    return max(-1.0, min(1.0, math.fsum(x*y for x, y in zip(a, b)) / denom))


def indexed(rows):
    result = {}
    for row in rows:
        sid = row["participant_id"]
        if not sid or sid != sid.strip() or sid in result:
            raise ValueError("Empty, padded or duplicate participant ID")
        result[sid] = row
    return result


def score(labels_path, submission_path):
    labels = indexed(read_table(labels_path, ["participant_id", "target", "split"]))
    if any(r["split"] not in ("train", "holdout") for r in labels.values()):
        raise ValueError("Unknown split")
    for row in labels.values():
        number(row["target"])
    truth = {sid: r for sid, r in labels.items() if r["split"] == "holdout"}
    submitted = indexed(read_table(submission_path, ["participant_id", "prediction", "half_a", "half_b"]))
    if set(truth) != set(submitted):
        raise ValueError("Submission IDs must exactly match holdout (no omissions, extras or train IDs)")
    ids = sorted(truth)
    target = [number(truth[s]["target"]) for s in ids]
    prediction = [number(submitted[s]["prediction"]) for s in ids]
    half_a = [number(submitted[s]["half_a"]) for s in ids]
    half_b = [number(submitted[s]["half_b"]) for s in ids]
    rho = correlation(ranks(target), ranks(prediction))
    reliability = correlation(half_a, half_b)
    return {"protocol": VERSION, "n_holdout": len(ids), "prediction_spearman": rho,
            "split_half_pearson": reliability,
            "status": "defined" if rho is not None and reliability is not None else "undefined_constant_vector",
            "labels_sha256": digest(labels_path), "submission_sha256": digest(submission_path),
            "rankable": False,
            "note": "Provisional diagnostics only. No combined score; halves require organizer code rerun."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--output", required=True)
    evaluate = sub.add_parser("score")
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--submission", required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.dataset, args.output) if args.command == "prepare" else score(args.labels, args.submission)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
