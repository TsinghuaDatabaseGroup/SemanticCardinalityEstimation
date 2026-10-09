"""AND threshold calibration for a trained judge; network weights and single-predicate thresholds are not changed.

For each model, its held-out training queries (val_queries in fold<k>/meta.json) are shuffled with
random.Random(split_seed): the first 2/3 form the calibration set, the last 1/3 the development set. Over all unordered
query pairs (a, b) of the calibration set, the joint score is min(score_a, score_b) and the joint truth is both teacher
labels = 1; on the 0.02-0.98 grid (step 0.01) the threshold whose total predicted count is closest to the total true
count is selected (the same count-matching objective as train.calibrate, applied to joint pairs). The development set is
only reported (pairwise q-error under the original vs the AND threshold) and does not affect the choice.
Requires a teacher label for every (query, document) pair in the snapshot; missing labels raise, nothing is imputed.
No evaluation truth is read, no teacher requests are made, nothing is retrained.

Artifact <model_dir>/and_calibration.json: {"folds": {fold: {original_threshold, and_threshold, calibration_query_ids,
development_query_ids, model_sha256, ...}}}.
"""
import itertools
import random
import time
from pathlib import Path

import numpy as np

from data import load_json, sha256_file, write_json
from judge import features, make_net

GRID = np.arange(0.02, 0.99, 0.01)


def pair_histogram(scores, labels, query_ids):
    """All query pairs of the calibration set: predicted joint counts per threshold (histogram accumulation, without
    materializing all pair rows) -> choose the threshold whose count ratio is closest to 1."""
    bins = np.zeros(len(GRID) + 1, dtype=np.int64)
    true_total = n_pairs = 0
    for a, b in itertools.combinations(query_ids, 2):
        joint = np.minimum(scores[a], scores[b])
        bins += np.bincount(np.searchsorted(GRID, joint, side="left"), minlength=len(bins))
        true_total += int(np.logical_and(labels[a], labels[b]).sum())
        n_pairs += 1
    counts = bins.sum() - np.cumsum(bins)[:-1]      # counts[i] = number of pairs with joint score > GRID[i]
    if true_total == 0:
        raise ValueError("no query pair in the calibration set has a joint positive (both teacher labels = 1)")
    choice = int(np.argmin(np.abs(counts / true_total - 1.0)))
    return float(GRID[choice]), {"n_query_pairs": n_pairs, "positive_teacher_pairs": true_total,
                                 "document_query_pairs": n_pairs * labels.shape[1], "selected_count": int(counts[choice]),
                                 "selected_count_ratio": float(counts[choice] / true_total)}


def development_counts(scores, labels, query_ids, thresholds):
    """All query pairs of the development set: per-pair q-error quantiles and pooled count ratio per threshold
    (report only)."""
    pairs = list(itertools.combinations(query_ids, 2))
    truth = np.array([np.logical_and(labels[a], labels[b]).sum() for a, b in pairs])
    out = {}
    for name, thr in thresholds.items():
        pred = np.array([(np.minimum(scores[a], scores[b]) > thr).sum() for a, b in pairs])
        err = np.maximum(np.maximum(pred, truth), 1) / np.maximum(np.minimum(pred, truth), 1)
        out[name] = {"threshold": thr, "n_pairs": len(pairs), "zero_teacher_count_pairs": int((truth == 0).sum()),
                     "p50": float(np.median(err)), "p90": float(np.percentile(err, 90)), "p95": float(np.percentile(err, 95)),
                     "p99": float(np.percentile(err, 99)), "max": float(err.max()),
                     "pooled_count_ratio": float(pred.sum() / max(truth.sum(), 1))}
    return out


def label_matrix(snapshot_dir, doc_ids):
    """Snapshot -> (query list, query vectors (n_q, dim), label matrix (n_q, n_docs) int8). Missing labels raise."""
    dj = load_json(Path(snapshot_dir) / "data_single.json")
    z = np.load(Path(snapshot_dir) / "query_vectors.npz", allow_pickle=False)
    if z["queries"].tolist() != [q["query"] for q in dj["queries"]]:
        raise ValueError("query_vectors.npz of the snapshot does not match the query list of data_single.json")
    labels = np.full((len(dj["queries"]), len(doc_ids)), -1, dtype=np.int8)
    row_of = {d: i for i, d in enumerate(doc_ids)}
    for qi, doc, _, y in dj["rows"]:
        if labels[qi, row_of[doc]] != -1:
            raise ValueError(f"duplicate (query {qi}, document {doc}) in the snapshot")
        labels[qi, row_of[doc]] = y
    missing = int((labels < 0).sum())
    if missing:
        raise ValueError(f"AND threshold calibration requires a teacher label for every (query, document) pair: "
                         f"{missing} missing ({labels.shape[0]} queries x {labels.shape[1]} documents)")
    return dj["queries"], z["Q"].astype(np.float32), labels


def fit(model_dir, snapshot_dir, doc_ids, emb, device="cpu", split_seed=42, log=print):
    import torch
    model_dir = Path(model_dir)
    out = model_dir / "and_calibration.json"
    if out.exists():
        raise SystemExit(f"{out} already exists: calibration results are not overwritten")
    top = load_json(model_dir / "meta.json")
    if top["snapshot"] != Path(snapshot_dir).name:
        raise SystemExit(f"the model was trained from snapshot {top['snapshot']}, not {Path(snapshot_dir).name}")
    t0 = time.time()
    _, qvec, labels = label_matrix(snapshot_dir, doc_ids)
    fold_dirs = {f: model_dir / f for f in top["folds"]} if top["protocol"] == "kfold" else {".": model_dir}
    report = {"method": __doc__, "model": str(model_dir), "snapshot": str(snapshot_dir), "split_seed": split_seed,
              "grid": [float(GRID[0]), float(GRID[-1]), 0.01], "device": device,
              "snapshot_sha256": sha256_file(Path(snapshot_dir) / "data_single.json"),
              "query_vectors_sha256": sha256_file(Path(snapshot_dir) / "query_vectors.npz"),
              "evaluation_truth_read": False, "teacher_calls": 0, "networks_retrained": False, "single_threshold_changed": False, "folds": {}}
    for name, d in fold_dirs.items():
        m = load_json(d / "meta.json")["single"]
        val = list(m["val_queries"])
        shuffled = list(val)
        random.Random(split_seed).shuffle(shuffled)
        n_cal = len(shuffled) * 2 // 3
        cal, dev = sorted(shuffled[:n_cal]), sorted(shuffled[n_cal:])
        net = make_net(emb.shape[1] * 4, m["hidden"], m["layers"], m["dropout"])
        net.load_state_dict(torch.load(d / "single.pt", map_location="cpu", weights_only=True))
        net.to(device).eval()
        scores = np.full(labels.shape, np.nan, dtype=np.float32)
        with torch.no_grad():
            for qi in val:
                x = features(np.broadcast_to(qvec[qi], emb.shape), emb)
                scores[qi] = net(torch.from_numpy(x).to(device)).view(-1).cpu().numpy()
        thr, cal_info = pair_histogram(scores, labels, cal)
        dev_info = development_counts(scores, labels, dev, {"original_threshold": m["threshold"], "and_threshold": thr})
        report["folds"][name] = {"original_threshold": m["threshold"], "and_threshold": thr,
                                 "calibration_query_ids": cal, "development_query_ids": dev,
                                 "model_sha256": sha256_file(d / "single.pt"), "calibration": cal_info, "development": dev_info,
                                 "offline_predicate_inferences": len(val) * len(doc_ids)}
        log(f"[{name}] single threshold {m['threshold']:.2f} -> AND threshold {thr:.2f} (calibration {len(cal)} queries, "
            f"{cal_info['n_query_pairs']} pairs, joint positives {cal_info['positive_teacher_pairs']}, "
            f"count ratio {cal_info['selected_count_ratio']:.3f}); "
            f"development p50 original {dev_info['original_threshold']['p50']:.3f} -> AND {dev_info['and_threshold']['p50']:.3f}")
        del net
    report["seconds"] = round(time.time() - t0, 1)
    write_json(out, report, indent=1)
    log(f"AND thresholds written to {out}")
    return report


def load_thresholds(model_dir, fold_names):
    """Read and_calibration.json, verify the model file hashes, and return {fold: and_threshold}."""
    model_dir = Path(model_dir)
    cal = load_json(model_dir / "and_calibration.json")
    out = {}
    for f in fold_names:
        if f not in cal["folds"]:
            raise SystemExit(f"and_calibration.json has no threshold for {f}")
        d = model_dir / f if f != "." else model_dir
        if sha256_file(d / "single.pt") != cal["folds"][f]["model_sha256"]:
            raise SystemExit(f"{f}/single.pt hash differs from the model file used for calibration")
        out[f] = float(cal["folds"][f]["and_threshold"])
    return out
