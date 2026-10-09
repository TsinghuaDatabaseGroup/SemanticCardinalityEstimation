"""Online estimation: node selection -> candidate set -> stratified importance sampling -> judging ->
bias-corrected extrapolation, for a whole evaluation batch.
Sampling plans for all queries are built first (verdicts do not affect sampling; single -> and -> or
consume one random-number stream in sequence) and saved. Judging is then grouped by document so that
all queries for one document are sent consecutively (server-side prefix cache). A (document, query)
pair drawn several times is judged once and weighted by its draws in extrapolation.
If the candidate set is empty (no selected node has documents), each selected node falls back to its
nearest ancestor with documents (root = whole corpus); the result is marked fallback with fallback_nodes.
With K fold models, the same plans are judged per fold and the per-query arithmetic mean of the fold
estimates is written to avg/. Per-fold verdicts are saved to verdicts_<set>.json.
With and_weight = "predicate_geomean", AND-set within-stratum weights use the geometric mean of the
per-predicate cosines (sample.predicate_weights); predicate vectors come from qvec_of(make_query([p], "single")).
"""
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import sha256_file
from judge import make_query
from sample import AND_WEIGHTS, build_sample_plan, extrapolate


def ancestors_with_docs(nodes, catalog, index_rows):
    """For each selected node, the nearest ancestor (including itself) that has documents; the root
    (= whole corpus) if none. Order-preserving dedup. With no selected nodes, returns [root]."""
    out = []
    for n in nodes:
        a = n
        while a is not None and not index_rows.get(a):
            a = catalog.parent.get(a)
        if a is None:
            a = catalog.root_name
        if a not in out:
            out.append(a)
    if not out:
        out.append(catalog.root_name)
    return out


def plan_query(query, selector, index_rows, emb, qvec, n_docs, budget, rng, catalog, pred_vecs=None):
    """If pred_vecs (predicate vector matrix) is not None, within-stratum weights use the geometric mean of predicate cosines."""
    nodes = selector.select(query["query"])
    node_lists = [index_rows.get(n, []) for n in nodes]
    fallback = not any(node_lists)
    fallback_nodes = None
    if fallback:
        fallback_nodes = ancestors_with_docs(nodes, catalog, index_rows)
        node_lists = [index_rows[a] if index_rows.get(a) else list(range(n_docs)) for a in fallback_nodes]
    groups, cand = build_sample_plan(node_lists, emb, qvec, budget, rng, pred_vecs=pred_vecs)
    return {"nodes": nodes, "fallback": fallback, "fallback_nodes": fallback_nodes, "cand_size": len(cand), "groups": groups}


def predicate_matrix(query, qvec_of):
    """Single-predicate query vectors of each distinct predicate of an AND query (first-seen order), shape (n_predicates, dim)."""
    return np.stack([qvec_of(make_query([p], "single")) for p in dict.fromkeys(query["predicates"])])


def judge_set(name, queries, connector, plans, judge, llm, log=print):
    t0 = time.time()
    by_row = defaultdict(set)
    for qi, p in enumerate(plans):
        for g in p["groups"]:
            for row in g["rows"]:
                by_row[row].add(qi)
    groups = [[(row, qi) for qi in sorted(qis)] for row, qis in sorted(by_row.items())]
    total = sum(len(g) for g in groups)
    draws = sum(len(g["rows"]) for p in plans for g in p["groups"])
    log(f"[{name}] {len(queries)} queries, {draws} draws, {total} judgements, {len(groups)} distinct documents")
    done, lock = [0], threading.Lock()

    def fn(item):
        row, qi = item
        v = judge.verdict(row, queries[qi], connector)
        with lock:
            done[0] += 1
            if done[0] % 500 == 0:
                log(f"  judged {done[0]}/{total}  {time.time() - t0:.0f}s")
        return qi, row, v

    verdicts = {}
    for res in llm.map_groups(groups, fn):
        for qi, row, v in res:
            verdicts[(qi, row)] = v
    rows = []
    for qi, (q, p) in enumerate(zip(queries, plans)):
        contribs = [[verdicts[(qi, r)] for r in g["rows"]] for g in p["groups"]]
        est = extrapolate(p["groups"], contribs)
        sampled = [r for g in p["groups"] for r in g["rows"]]
        rows.append({"case_id": f"{name}:{qi:03d}", "query": q["query"],
                     "predicates": list(q["predicates"]), "estimate": float(est),
                     "cand_size": p["cand_size"], "sample_size": len(sampled),
                     "n_unique": len(set(sampled)), "n_strata": len(p["groups"]),
                     "nodes": p["nodes"], "fallback": p["fallback"], "fallback_nodes": p["fallback_nodes"]})
    log(f"[{name}] elapsed {time.time() - t0:.0f}s; {total} judgements, {draws} draws")
    return rows, verdicts


def save_rows(path, rows, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


def run(queries_by_set, selector, index_rows, emb, qvec_of, doc_ids, judges, llm, budget, rng, out, meta, catalog, log=print,
        and_weight="query"):
    """judges = {"fold0": judge, ...} or {".": judge} (single model / direct LLM judge). Returns the result
    directory (avg/ or out). and_weight: within-stratum weight mode for the AND set (sample.AND_WEIGHTS)."""
    if and_weight not in AND_WEIGHTS:
        raise ValueError(f"and_weight must be one of {AND_WEIGHTS}, not {and_weight!r}")
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} already exists: evaluation batches are never overwritten, choose another name")
    plans, calls0, hits0 = {}, llm.calls, llm.hits
    for s, qs in queries_by_set.items():
        geo = s == "and" and and_weight == "predicate_geomean"
        plans[s] = [plan_query(q, selector, index_rows, emb, qvec_of(q["query"]), len(doc_ids), budget, rng, catalog,
                               pred_vecs=predicate_matrix(q, qvec_of) if geo else None) for q in qs]
        save_rows(out / f"plans_{s}.json",
                  [{"query": q["query"], "nodes": p["nodes"], "fallback": p["fallback"], "fallback_nodes": p["fallback_nodes"],
                    "cand_size": p["cand_size"],
                    "groups": [{"docs": [doc_ids[r] for r in g["rows"]], "w": g["w"], "floored": g.get("floored", 0)} for g in p["groups"]]}
                   for q, p in zip(qs, plans[s])], {**meta, "what": "Sampling plan: node order, documents drawn per stratum with extrapolation weights; fallback rows record the ancestor nodes used"})
    plan_stats = {s: {"rows": len(ps), "draws": sum(len(g["rows"]) for p in ps for g in p["groups"]),
                      "fallback_rows": sum(p["fallback"] for p in ps),
                      "floored_docs": sum(g.get("floored", 0) for p in ps for g in p["groups"])} for s, ps in plans.items()}
    select_calls, select_hits = llm.calls - calls0, llm.hits - hits0
    per_fold = {}
    for name, judge in judges.items():
        sub = out if name == "." else out / name
        per_fold[name] = {}
        for s, qs in queries_by_set.items():
            c0, h0 = llm.calls, llm.hits
            rows, verdicts = judge_set(s, qs, s, plans[s], judge, llm, log)
            per_fold[name][s] = rows
            save_rows(sub / f"estimates_{s}.json", rows,
                      {**meta, "model": name, "judge_calls": judge.calls, "judge_unparsed": judge.unparsed,
                       "judge_format_repairs": getattr(judge, "format_repairs", None),
                       "judge_sub_calls": getattr(judge, "sub_calls", None), "llm_calls": llm.calls, "llm_cache_hits": llm.hits,
                       "llm_new_calls_this_set": llm.calls - c0, "llm_cache_hits_this_set": llm.hits - h0})
            (sub / f"verdicts_{s}.json").write_text(json.dumps([[qi, doc_ids[r], int(v)] for (qi, r), v in sorted(verdicts.items())]),
                                                    encoding="utf-8")
    result = out
    if len(judges) > 1:                                   # K folds: average the estimates per query
        result = out / "avg"
        names = list(judges)
        for s in queries_by_set:
            rows = [dict(r0, estimate=float(np.mean([per_fold[n][s][i]["estimate"] for n in names])))
                    for i, r0 in enumerate(per_fold[names[0]][s])]
            save_rows(result / f"estimates_{s}.json", rows, {**meta, "model": "mean of " + " ".join(names)})
    (out / "meta.json").write_text(json.dumps({**meta, "models": list(judges), "result": str(result.relative_to(out)),
                                              "and_weight": and_weight,
                                              "sets": {s: len(qs) for s, qs in queries_by_set.items()},
                                              "plans": plan_stats, "plan_sha256": {s: sha256_file(out / f"plans_{s}.json") for s in queries_by_set},
                                              "select_llm_new_calls": select_calls, "select_llm_cache_hits": select_hits,
                                              "llm_new_calls_total": llm.calls, "llm_cache_hits_total": llm.hits},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    return result
