"""Train the single-predicate dual-MLP judge from a snapshot (data_single.json + query_vectors.npz).

- Features: unit vectors of query and document, concatenated as [q, d, q*d, |q-d|] into an ln_res MLP.
- Split by whole queries: the query order is shuffled with random.Random(17); kfold holds out part k of K for model k
  (estimate.py averages the K models' estimates per query); split holds out the first val_frac of the queries.
- Loss: asymmetric binary cross-entropy (negative term weighted by neg_weight); Adam; the epoch with the best
  validation AUC is kept (ties go to the later epoch; validation loss is used when the validation part has one class).
- Threshold: on validation rows with source == cand, pick the grid point (0.02-0.98, step 0.01) where TPR + r*FPR is
  closest to 1 (r = negative/positive ratio); fall back to all rows with fewer than 50 cand rows or one class.
  Positive means p > threshold. Validation forward passes run in batches and are concatenated on the CPU.
Artifacts: <out>/fold<k>/{single.pt, meta.json} (split writes directly into <out>/), <out>/meta.json with the protocol,
the snapshot and per-fold thresholds.
"""
import json
import random
import time
from pathlib import Path

import numpy as np

from judge import features, make_net

VAL_SEED = 17


def auc(scores, y):
    y = np.asarray(y, bool)
    if not y.any() or y.all():
        return float("nan")
    r = np.empty(len(scores))
    r[np.argsort(scores)] = np.arange(1, len(scores) + 1)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def calibrate(scores, y):
    """Count-unbiased threshold: TPR + r*FPR closest to 1 (r = negative/positive ratio)."""
    s, y = np.asarray(scores), np.asarray(y, bool)
    P, N = s[y], s[~y]
    if not len(P) or not len(N):
        return 0.5, {}
    r = len(N) / len(P)
    ts = np.arange(0.02, 0.99, 0.01)
    bias = np.array([(P > t).mean() + r * (N > t).mean() for t in ts])
    i = int(np.argmin(np.abs(bias - 1.0)))
    return float(ts[i]), {"tpr": float((P > ts[i]).mean()), "fpr": float((N > ts[i]).mean()),
                          "neg_pos_ratio": r, "count_bias": float(bias[i])}


def calibrate_on_cand(scores, y, sources, min_rows=50):
    """Use only cand rows (the distribution seen at estimation time) when there are >= min_rows of them with both
    classes; otherwise fall back to all rows."""
    keep = np.array([s == "cand" for s in sources])
    yk = np.asarray(y)[keep]
    if keep.sum() >= min_rows and 0 < yk.sum() < keep.sum():
        t, info = calibrate(np.asarray(scores)[keep], yk)
        return t, {**info, "calibrated_on": f"{int(keep.sum())} cand rows"}
    t, info = calibrate(scores, y)
    return t, {**info, "calibrated_on": "all rows"}


def fit(Q, D, qi, row, y, source, hp, device, is_val, log):
    """Q: query unit vectors (n_q, dim); D: document unit vectors; qi/row/y/source: one entry per pair; is_val marks
    the validation rows."""
    import torch
    qi, row, y = np.asarray(qi), np.asarray(row), np.asarray(y, np.float32)
    is_val = np.asarray(is_val, bool)
    va, tr = np.flatnonzero(is_val).astype(np.int64), np.flatnonzero(~is_val).astype(np.int64)
    log(f"  train {len(tr)} pairs ({len(set(qi[tr].tolist()))} queries) / validation {len(va)} pairs ({len(set(qi[va].tolist()))} queries), "
        f"positive rate {y.mean():.1%}, {hp['epochs']} epochs; validation batch size {hp['batch_size']}")

    def X(idx):
        return torch.from_numpy(features(Q[qi[idx]], D[row[idx]])).to(device)

    def bce(p, t, w=1.0):
        p = p.clamp(1e-6, 1 - 1e-6)
        return -(t * torch.log(p) + w * (1 - t) * torch.log(1 - p)).mean()

    def validate(net):
        """Batched validation forward pass, concatenated on the CPU in the original row order."""
        chunks = []
        with torch.no_grad():
            for start in range(0, len(va), hp["batch_size"]):
                idx = va[start:start + hp["batch_size"]]
                chunks.append(net(X(idx)).view(-1).cpu())
        return torch.cat(chunks)

    torch.manual_seed(hp["seed"])
    net = make_net(Q.shape[1] * 4, hp["hidden"], hp["layers"], hp["dropout"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    gen = torch.Generator().manual_seed(hp["seed"])
    yv = torch.from_numpy(y[va])
    best = (-1e9, None, -1, float("nan"))
    for ep in range(hp["epochs"]):
        net.train()
        for sl in torch.randperm(len(tr), generator=gen).split(hp["batch_size"]):
            idx = tr[sl.numpy()]
            loss = bce(net(X(idx)).view(-1), torch.from_numpy(y[idx]).to(device), hp["neg_weight"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        pv = validate(net)
        a = auc(pv.numpy(), y[va])
        crit = a if a == a else -bce(pv, yv).item()      # one-class validation part: select the epoch by validation loss
        if crit >= best[0]:                                   # ties go to the later epoch
            best = (crit, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep, a)
        if (ep + 1) % 10 == 0 or ep + 1 == hp["epochs"]:
            log(f"  epoch {ep + 1}/{hp['epochs']}  loss {loss.item():.4f}  val AUC {a:.4f}")
    net.load_state_dict(best[1])
    net.eval()
    pv = validate(net).numpy()
    thr, info = calibrate_on_cand(pv, y[va], [source[i] for i in va])
    return net.cpu(), thr, {**info, "val_auc": best[3], "best_epoch": best[2],
                            "n_train": int(len(tr)), "n_val": int(len(va)),
                            "validation_batch_size": hp["batch_size"],
                            "validation_implementation": "row batches; CPU score concatenation in original order"}


def save(out, net, thr, hp, dj_meta, info, extra, t0):
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "single.pt")
    meta = {"single": {"hidden": hp["hidden"], "layers": hp["layers"], "dropout": hp["dropout"], "threshold": thr,
                       **dj_meta, **info, "lr": hp["lr"], "epochs": hp["epochs"], "neg_weight": hp["neg_weight"],
                       "batch_size": hp["batch_size"], "seed": hp["seed"], **extra,
                       "train_seconds": round(time.time() - t0)}}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta["single"]


def run(hp, snapshot_dir, out, doc_ids, emb, protocol="kfold", k=5, device="cpu", log=print):
    snapshot_dir, out = Path(snapshot_dir), Path(out)
    if out.exists():
        raise SystemExit(f"{out} already exists: model directories are not overwritten; choose another name")
    dj = json.loads((snapshot_dir / "data_single.json").read_text(encoding="utf-8"))
    z = np.load(snapshot_dir / "query_vectors.npz", allow_pickle=False)
    qs, rows = dj["queries"], dj["rows"]
    if z["queries"].tolist() != [q["query"] for q in qs]:
        raise SystemExit("query_vectors.npz of the snapshot does not match the query list of data_single.json")
    Q = z["Q"].astype(np.float32)
    row_of = {d: i for i, d in enumerate(doc_ids)}
    qi, row = [r[0] for r in rows], [row_of[r[1]] for r in rows]
    src, y = [r[2] for r in rows], [r[3] for r in rows]
    order = list(range(len(qs)))
    random.Random(VAL_SEED).shuffle(order)
    dj_meta = {"data": str(snapshot_dir / "data_single.json"), **dj["meta"]}
    log(f"{protocol}: {snapshot_dir.name}, {len(qs)} queries, {len(rows)} pairs ({dj['meta']['llm']}); {emb.shape[1]}-dim vectors; "
        f"device {device}; output {out}")
    top = {"protocol": protocol, "snapshot": snapshot_dir.name, "n_queries": len(qs), "pairs": len(rows),
           "embedding_dim": int(emb.shape[1]), "device": device, "val_seed": VAL_SEED,
           "hp": {k_: hp[k_] for k_ in ("hidden", "layers", "dropout", "lr", "epochs", "batch_size", "neg_weight", "seed")}}
    folds = {}
    if protocol == "split":
        n_val = max(1, int(len(qs) * hp["val_frac"]))
        t0 = time.time()
        net, thr, info = fit(Q, emb, qi, row, y, src, hp, device, np.isin(qi, order[:n_val]), log)
        folds["."] = save(out, net, thr, hp, dj_meta, info,
                          {"protocol": "split: 15% of queries held out for validation", "val_queries": sorted(order[:n_val])}, t0)
        log(f"[split] threshold {thr:.2f}  val AUC {info['val_auc']:.4f} (epoch {info['best_epoch'] + 1})  calibrated on {info['calibrated_on']}")
    elif protocol == "kfold":
        top["k"] = k
        for f in range(k):
            val_q = sorted(order[f * len(qs) // k:(f + 1) * len(qs) // k])
            t0 = time.time()
            log(f"[fold {f}/{k}]")
            net, thr, info = fit(Q, emb, qi, row, y, src, hp, device, np.isin(qi, val_q), log)
            folds[f"fold{f}"] = save(out / f"fold{f}", net, thr, hp, dj_meta, info,
                                     {"protocol": f"kfold: fold {f} of {k} held out", "fold": f, "k": k, "val_queries": val_q}, t0)
            log(f"[fold {f}] threshold {thr:.2f}  val AUC {info['val_auc']:.4f} (epoch {info['best_epoch'] + 1})  "
                f"calibrated on {info['calibrated_on']}  TPR {info.get('tpr', float('nan')):.3f} FPR {info.get('fpr', float('nan')):.3f}")
    else:
        raise SystemExit(f"unknown protocol {protocol} (kfold / split)")
    top["folds"] = {name: {"threshold": m["threshold"], "val_auc": m["val_auc"], "best_epoch": m["best_epoch"],
                           "calibrated_on": m["calibrated_on"], "n_train": m["n_train"], "n_val": m["n_val"]}
                    for name, m in folds.items()}
    if "." in folds:                                      # split: the model meta and the top-level meta share one file
        top["single"] = folds["."]
    (out / "meta.json").write_text(json.dumps(top, ensure_ascii=False, indent=1), encoding="utf-8")
    return top
