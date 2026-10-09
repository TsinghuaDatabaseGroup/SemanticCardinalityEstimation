"""End-to-end evaluation: estimates + ground truth -> q-error table. Only this module reads labels.jsonl.
Ground truth is computed per predicate from the document labels (single: predicate in the label set;
AND: all predicates present, i.e. set intersection; OR: any present).
qerr = max(true, est) / max(1, min(true, est)); quantiles use np.quantile linear interpolation;
reports n / p50 / p90 / p95 / p99 / max / mean / geo.
Two views: all rows; deduplicated by query text (first occurrence kept within each set).
Writes <dir>/metrics.json, per_query.json, report.txt.
"""
import json
from pathlib import Path

import numpy as np

from data import load_labels, load_queries, write_json

QS = (0.50, 0.90, 0.95, 0.99)


def qerr(est, true):
    return max(true, est) / max(1.0, min(true, est))


def truth_count(predicates, connector, tags_by_doc):
    preds = list(predicates)
    n = 0
    for tags in tags_by_doc.values():
        if connector == "single":
            hit = preds[0] in tags
        elif connector == "and":
            hit = all(p in tags for p in preds)
        else:
            hit = any(p in tags for p in preds)
        n += hit
    return n


def stats(v):
    v = np.asarray(v, dtype=float)
    q = np.quantile(v, QS)
    return {"n": int(len(v)), "p50": float(q[0]), "p90": float(q[1]),
            "p95": float(q[2]), "p99": float(q[3]), "max": float(v.max()),
            "mean": float(v.mean()), "geo": float(np.exp(np.log(v).mean()))}


def build_table(by_set):
    groups = {}
    if "single" in by_set:
        groups["Single-predicate"] = by_set["single"]
    multi = [by_set[k] for k in ("and", "or") if k in by_set]
    if multi:
        groups["Multi-predicate (AND+OR)"] = np.concatenate(multi)
    for k, label in (("and", "  - AND only"), ("or", "  - OR only")):
        if k in by_set:
            groups[label] = by_set[k]
    if len(by_set) > 1:
        groups["Overall"] = np.concatenate(list(by_set.values()))
    return {label: stats(v) for label, v in groups.items()}


def render(table, label=""):
    hdr = (f"{'query set':<26s}{'n':>5s}{'p50':>8s}{'p90':>8s}{'p95':>8s}"
           f"{'p99':>8s}{'max':>9s}{'mean':>8s}{'geo':>8s}")
    lines = ([f"########## {label}"] if label else []) + [hdr, "-" * len(hdr)]
    for name, s in table.items():
        lines.append(f"{name:<26s}{s['n']:>5d}{s['p50']:>8.3f}{s['p90']:>8.3f}"
                     f"{s['p95']:>8.3f}{s['p99']:>8.3f}{s['max']:>9.3f}"
                     f"{s['mean']:>8.3f}{s['geo']:>8.3f}")
    return "\n".join(lines)


def rows_with_truth(result_dir, set_name, query_path, tags_by_doc):
    rows = json.loads((Path(result_dir) / f"estimates_{set_name}.json").read_text(encoding="utf-8"))["rows"]
    queries = load_queries(query_path)
    if len(rows) > len(queries):
        raise ValueError(f"{set_name}: {len(rows)} estimate rows exceed {len(queries)} query rows")
    for r, q in zip(rows, queries):
        if r["query"] != q["query"]:
            raise ValueError(f"{set_name}: estimate rows and query file are not in the same order: {r['case_id']}")
        r["true"] = float(truth_count(q["predicates"], set_name, tags_by_doc))
        r["qerr"] = qerr(r["estimate"], r["true"])
    return rows


def dedup(rows):
    seen, out = set(), []
    for r in rows:
        if r["query"] not in seen:
            seen.add(r["query"])
            out.append(r)
    return out


def report(result_dir, sets, query_paths, labels_path, doc_ids, label="", out_dir=None):
    """Ground truth counts only documents in the current corpus; out_dir defaults to result_dir.
    Returns {set: rows with true/qerr}."""
    tags = load_labels(labels_path)
    missing = [d for d in doc_ids if d not in tags]
    if missing:
        raise ValueError(f"Labels are missing for {len(missing)} corpus documents: {missing[:3]}")
    tags = {d: tags[d] for d in doc_ids}
    rows_by_set = {s: rows_with_truth(result_dir, s, query_paths[s], tags) for s in sets}
    tables = {"all_rows": build_table({s: [r["qerr"] for r in rows] for s, rows in rows_by_set.items()}),
              "dedup": build_table({s: [r["qerr"] for r in dedup(rows)] for s, rows in rows_by_set.items()})}
    worst = sorted((r for rows in rows_by_set.values() for r in rows), key=lambda r: -r["qerr"])[:8]
    text = "\n\n".join([render(tables["all_rows"], f"{label} -- all rows"),
                        render(tables["dedup"], f"{label} -- deduplicated by query text"),
                        "Worst 8:\n" + "\n".join(
                            f"  q={r['qerr']:7.2f}  true {r['true']:7.0f}  est {r['estimate']:8.1f}  "
                            f"cand {r['cand_size']:5d}  {r['query']}" for r in worst)])
    print(text)
    out_dir = Path(out_dir or result_dir)
    write_json(out_dir / "metrics.json", {"label": label, "labels": str(labels_path), **tables}, indent=1)
    write_json(out_dir / "per_query.json", rows_by_set, indent=1)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    return rows_by_set
