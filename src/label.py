"""Resumable teacher labeling of single-predicate training queries, written in blocks; snapshot() freezes an export.

Each query draws documents from a deterministic stream (candidate set / nearest neighbours / uniform, refilled in that
order), optionally preceded by a positive-seeking stream; each block serves the queries with the fewest labeled pairs.
An optional time budget (--minutes) is persisted as stages in budgets.json. Stop with `touch <out>/STOP`.

Artifacts in <out>/: queries.json, selections.json, stream_inputs.npz, chunks/chunk_NNNNN.json
({"meta", "rows": [[qi, doc_id, source, 0/1]]}), data_single.json ({"meta", "queries", "rows"}, input of train.py),
progress.json, budgets.json, format_repairs.jsonl.
"""
import hashlib
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import write_json
from judge import Judge
from selector import Selector

CAND_FRAC, NEAR_FRAC = 0.60, 0.25


def now():
    """Wall clock (replaceable in tests)."""
    return time.time()


def quota(k):
    n_cand, n_near = int(k * CAND_FRAC), int(k * NEAR_FRAC)
    return (("cand", n_cand), ("near", n_near), ("uni", k - n_cand - n_near))


class Streams:
    """Deterministic per-query order over three pools; take(qi, done) returns the next round as (row, source) pairs,
    skipping done; take_seek is the positive-seeking stream."""

    def __init__(self, queries, cand_rows, near_rows, n_docs, per_round, seed, sims_fn=None, tag_rows=None):
        self.queries, self.cand, self.near = queries, cand_rows, near_rows
        self.n, self.quota, self.seed, self.per_round = n_docs, quota(per_round), seed, per_round
        self.sims_fn, self.tag_rows = sims_fn, tag_rows or [[] for _ in queries]

    def take_seek(self, qi, done):
        """Seek stream: same-tag documents (cosine descending), then the whole corpus by cosine descending; skips done."""
        sims = self.sims_fn(qi)
        seen, picked = set(done), []
        for src, rows in (("seek_tag", sorted(self.tag_rows[qi], key=lambda r: (-sims[r], r))),
                          ("seek_top", np.argsort(-sims, kind="stable"))):
            for r in rows:
                if len(picked) >= self.per_round:
                    return picked
                r = int(r)
                if r not in seen:
                    seen.add(r)
                    picked.append((r, src))
        return picked

    def orders(self, qi):
        h = hashlib.sha256(f"{self.seed}|{self.queries[qi]['query']}".encode("utf-8")).hexdigest()
        rng = np.random.default_rng(int(h[:16], 16))
        return {"cand": rng.permutation(np.asarray(self.cand[qi], dtype=np.int64)),
                "near": rng.permutation(np.asarray(self.near[qi], dtype=np.int64)),
                "uni": rng.permutation(self.n)}

    def take(self, qi, done):
        orders, seen, picked = self.orders(qi), set(done), []
        ptr = dict.fromkeys(orders, 0)

        def pull(src, k):
            got, arr = 0, orders[src]
            while got < k and ptr[src] < len(arr):
                r = int(arr[ptr[src]])
                ptr[src] += 1
                if r not in seen:
                    seen.add(r)
                    picked.append((r, src))
                    got += 1
            return got

        short = sum(k - pull(src, k) for src, k in self.quota)
        for src, _ in self.quota:                         # refill the shortfall from cand -> near -> uni
            if short <= 0:
                break
            short -= pull(src, short)
        return picked


class SeekPolicy:
    def __init__(self, target, cap, min_rounds):
        self.target, self.cap, self.min_rounds = target, cap, min_rounds

    def active(self, st):
        return (st["pos"] < self.target and st["seek_pairs"] < self.cap
                and not (st["seek_rounds"] >= self.min_rounds and st["last_seek_pos"] == 0))


def new_stats(n):
    return [{"pos": 0, "seek_pairs": 0, "seek_rounds": 0, "last_seek_pos": None} for _ in range(n)]


def absorb(stats, rows):
    """Add one block's rows to the stats; the seek rows of one query within a block count as one seek round."""
    seek = {}
    for qi, _, src, y in rows:
        stats[qi]["pos"] += int(y)
        if str(src).startswith("seek"):
            n, p = seek.get(qi, (0, 0))
            seek[qi] = (n + 1, p + int(y))
    for qi, (n, p) in seek.items():
        st = stats[qi]
        st["seek_pairs"] += n
        st["seek_rounds"] += 1
        st["last_seek_pos"] = p


def chunk_files(out):
    return sorted((Path(out) / "chunks").glob("chunk_*.json"))


def export(out, queries, all_rows, meta):
    ys = [r[3] for r in all_rows]
    write_json(Path(out) / "data_single.json", {
        "meta": {**meta, "n_queries": len(queries), "pairs": len(all_rows), "chunks": len(chunk_files(out)),
                 "pos_rate": float(np.mean(ys)) if ys else 0.0, "updated": time.strftime("%F %T")},
        "queries": queries, "rows": all_rows})


def progress(out, queries, all_rows, done, n_chunks, last, stats, policy):
    counts = np.array([len(s) for s in done])
    pos = np.zeros(len(queries), int)
    for qi, _, _, y in all_rows:
        pos[qi] += y
    write_json(Path(out) / "progress.json", {
        "updated": time.strftime("%F %T"), "chunks": n_chunks, "pairs": len(all_rows),
        "pos_rate": float(np.mean([r[3] for r in all_rows])) if all_rows else 0.0,
        "pairs_per_query": {"min": int(counts.min()), "median": float(np.median(counts)), "max": int(counts.max())},
        "queries_with_pos": int((pos > 0).sum()), "queries_with_pos_ge3": int((pos >= 3).sum()),
        "queries_with_pos_ge10": int((pos >= 10).sum()), "n_queries": len(queries),
        "all_pairs": int(len(queries) * last["n_docs"]),
        "seek_pairs": int(sum(st["seek_pairs"] for st in stats)),
        "seek_pos": int(sum(1 for r in all_rows if str(r[2]).startswith("seek") and r[3])),
        "seeking_active": int(sum(policy.active(st) for st in stats)) if policy else 0,
        "last_chunk": last})


def label_loop(queries, streams, doc_ids, label_fn, out, chunk_pairs, meta, log=print, policy=None, max_chunks=0,
               deadline=None):
    """Main loop. label_fn(qis, pairs) -> 0/1 labels shaped like pairs (None for verdicts not started before the
    deadline; those are dropped). policy = SeekPolicy enables positive seeking before the regular rounds.
    max_chunks > 0 limits the blocks written in this call; deadline is checked before each block (reason TIME_BUDGET).
    data_single.json is re-exported after every block. Returns (exit reason, blocks written)."""
    out = Path(out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    row_of = {d: i for i, d in enumerate(doc_ids)}
    all_rows, done, stats = [], [set() for _ in queries], new_stats(len(queries))
    for f in chunk_files(out):
        kept = []
        for qi, d, src, y in json.loads(f.read_text(encoding="utf-8"))["rows"]:
            if row_of[d] not in done[qi]:
                done[qi].add(row_of[d])
                kept.append([qi, d, src, y])
        absorb(stats, kept)
        all_rows += kept
    next_id = 1 + max([int(f.stem.split("_")[1]) for f in chunk_files(out)] or [-1])
    n_docs, written, reason = len(doc_ids), 0, None
    log(f"Resuming: {len(chunk_files(out))} blocks, {len(all_rows)} pairs so far; next block {next_id}")
    while True:
        if (out / "STOP").exists():
            reason = "STOP"
            break
        if max_chunks and written >= max_chunks:
            reason = "MAX_CHUNKS"
            break
        if deadline is not None and now() >= deadline:
            reason = "TIME_BUDGET"
            break
        order = sorted((qi for qi in range(len(queries)) if len(done[qi]) < n_docs),
                       key=lambda qi: (len(done[qi]), qi))
        if not order:
            reason = "ALL_DONE"
            break
        qis, pairs, total = [], [], 0
        seeking = sorted((qi for qi in order if policy and policy.active(stats[qi])),
                         key=lambda qi: (stats[qi]["pos"], stats[qi]["seek_pairs"], qi))
        for qi in seeking:
            if total >= chunk_pairs:
                break
            ps = streams.take_seek(qi, done[qi])
            if ps:
                qis.append(qi)
                pairs.append(ps)
                total += len(ps)
        in_chunk = set(qis)
        for qi in order:                                   # balanced rotation (or fill up the block)
            if total >= chunk_pairs:
                break
            if qi in in_chunk:
                continue
            ps = streams.take(qi, done[qi])
            if ps:
                qis.append(qi)
                pairs.append(ps)
                total += len(ps)
        t0 = time.time()
        labels = label_fn(qis, pairs)
        rows = [[qi, doc_ids[r], src, int(v)] for qi, ps, ls in zip(qis, pairs, labels)
                for (r, src), v in zip(ps, ls) if v is not None]
        if not rows:                                       # deadline hit: nothing judged in this block
            reason = "TIME_BUDGET"
            break
        secs = time.time() - t0
        n_seek = sum(1 for r in rows if str(r[2]).startswith("seek"))
        last = {"chunk": next_id, "pairs": len(rows), "queries": len({r[0] for r in rows}),
                "pos_rate": float(np.mean([r[3] for r in rows])) if rows else 0.0,
                "seek_pairs": n_seek, "seek_pos": sum(r[3] for r in rows if str(r[2]).startswith("seek")),
                "seconds": round(secs), "pairs_per_s": round(len(rows) / max(secs, 1e-9), 2),
                "min_pairs_before": int(len(done[qis[0]])), "n_docs": n_docs, "time": time.strftime("%F %T"),
                "partial": len(rows) < total}
        write_json(out / "chunks" / f"chunk_{next_id:05d}.json", {"meta": last, "rows": rows})
        next_id += 1
        written += 1
        all_rows += rows
        for qi, d, _, _ in rows:
            done[qi].add(row_of[d])
        absorb(stats, rows)
        export(out, queries, all_rows, meta)
        progress(out, queries, all_rows, done, next_id, last, stats, policy)
        n_active = sum(policy.active(st) for st in stats) if policy else 0
        log(f"Block {last['chunk']}: {last['queries']} queries, {len(rows)} pairs (seek {n_seek}, seek positives {last['seek_pos']})"
            f"{', partial' if last['partial'] else ''}; positive rate {last['pos_rate']:.1%}; {last['seconds']}s ({last['pairs_per_s']} pairs/s); "
            f"total {len(all_rows)} pairs, min per query {min(len(s) for s in done)}; queries with positives {sum(st['pos'] > 0 for st in stats)}, "
            f">=10 positives {sum(st['pos'] >= 10 for st in stats)}, still seeking {n_active}")
        if last["partial"]:                                # a partial block means the deadline was hit
            reason = "TIME_BUDGET"
            break
    if not (out / "data_single.json").exists():
        export(out, queries, all_rows, meta)
    log(f"Exit: {reason}; total {len(all_rows)} pairs")
    return reason, written


def label_pairs(queries, pairs, judge, llm, log=print, deadline=None):
    """Teacher labels shaped like pairs (None for verdicts not started before the deadline). Requests are grouped by
    document so that the queries on one document are sent consecutively (server-side prefix cache)."""
    by_row = defaultdict(list)
    for qi, ps in enumerate(pairs):
        for k, (r, _) in enumerate(ps):
            by_row[r].append((qi, k))
    groups = [[(r, qi, k) for qi, k in items] for r, items in sorted(by_row.items())]
    total, done, t0, lock = sum(map(len, groups)), [0], time.time(), threading.Lock()

    def fn(it):
        r, qi, k = it
        if deadline is not None and now() >= deadline:
            return qi, k, None
        v = judge.verdict(r, queries[qi], "single")
        with lock:
            done[0] += 1
            if done[0] % 2000 == 0:
                log(f"  labeled {done[0]}/{total}  {time.time() - t0:.0f}s")
        return qi, k, v

    labels = [[None] * len(ps) for ps in pairs]
    for res in llm.map_groups(groups, fn):
        for qi, k, v in res:
            labels[qi][k] = v
    return labels


def load_stream_inputs(out, queries, qvecs, log=print):
    """Query vectors: read <out>/stream_inputs.npz if it exists and matches the queries; otherwise take them from the
    vector table (computing missing ones) and save them."""
    path = Path(out) / "stream_inputs.npz"
    texts = [q["query"] for q in queries]
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if z["queries"].tolist() == texts:
            log(f"Query vectors read from {path.name}")
            return z["Q"]
        raise SystemExit(f"{path} has a different query list; use another --run or delete the old directory")
    Q = qvecs.matrix(texts)
    write_npz(path, queries=np.array(texts), Q=Q)
    log(f"{len(texts)} query vectors saved to {path.name}")
    return Q


def write_npz(path, **arrays):
    import os
    tmp = Path(path).with_name(Path(path).name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def open_stage(out, minutes, chunks, pairs, log):
    """Stage budget: returns (budgets list, current stage dict or None). minutes <= 0 records no stage (no time limit)."""
    bpath = Path(out) / "budgets.json"
    budgets = json.loads(bpath.read_text(encoding="utf-8")) if bpath.exists() else []
    if minutes <= 0:
        return budgets, None
    if budgets and "finished_epoch" not in budgets[-1]:     # previous stage unfinished: reuse its deadline
        st = budgets[-1]
        log(f"Reusing the deadline of stage {st['stage']} ({(st['deadline_epoch'] - now()) / 60:.1f} minutes left; "
            "a restart does not reset the clock)")
        return budgets, st
    started = now()
    st = {"stage": len(budgets) + 1, "requested_minutes": float(minutes), "started_epoch": started,
          "deadline_epoch": started + float(minutes) * 60.0, "chunks_before": chunks, "pairs_before": pairs,
          "rule": "started after node selection and query-vector preparation; no new verdicts start after the deadline; "
                  "in-flight verdicts finish and the final partial block is kept; a restart keeps the same deadline; "
                  "more time means a new stage", "runs": []}
    budgets.append(st)
    write_json(bpath, budgets, indent=1)
    log(f"Stage {st['stage']}: {minutes} minutes, deadline {time.strftime('%F %T', time.localtime(st['deadline_epoch']))}")
    return budgets, st


def run(cfg, ds, queries, catalog, index_rows, doc_ids, texts, emb, qvecs, llm, per_doc, out, chunk_chars,
        select_cfg, max_chunks=0, minutes=0, log=print):
    """cfg = label section of the config; ds = dataset section (teacher prompt wording); per_doc = the three doc_label
    class names of each document (used for positive seeking); None or seek.target = 0 disables seeking.
    minutes > 0 opens (or resumes) a time-budget stage."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    qpath = out / "queries.json"
    if qpath.exists() and [q["query"] for q in json.loads(qpath.read_text(encoding="utf-8"))] != [q["query"] for q in queries]:
        raise SystemExit(f"{qpath} does not match the current query list; use another --run or delete the old directory")
    write_json(qpath, queries)
    log(f"{len(queries)} single-predicate queries (index-derived {sum(q['origin'] == 'index' for q in queries)}); "
        f"corpus {len(doc_ids)} documents; all pairs {len(queries) * len(doc_ids)}")

    spath = out / "selections.json"                      # frozen node selections: resuming does not ask the model again
    cached = json.loads(spath.read_text(encoding="utf-8")) if spath.exists() else {}
    selector = Selector(catalog, llm=llm, votes=select_cfg["votes"], temperature=select_cfg["temperature"],
                        cached=cached, log=log)
    t0 = time.time()
    llm.map_groups([[q["query"]] for q in queries], selector.select)
    write_json(spath, {q["query"]: selector.select(q["query"]) for q in queries})
    cand = [sorted({r for n in selector.select(q["query"]) for r in index_rows.get(n, [])}) for q in queries]
    log(f"Node selection done in {time.time() - t0:.0f}s (LLM new calls {llm.calls}, cache hits {llm.hits}); "
        f"median candidate set {int(np.median([len(c) for c in cand]))} documents, empty candidate sets {sum(not c for c in cand)}")

    Q = load_stream_inputs(out, queries, qvecs, log)
    near = np.argsort(-(emb @ Q.T), axis=0)[:cfg["near_pool"]].T.tolist()
    seek = cfg.get("seek") or {}
    policy = SeekPolicy(seek["target"], seek["cap"], seek["min_rounds"]) if seek.get("target") else None
    tag_rows = [[] for _ in queries]
    if policy and per_doc:
        row_of = {d: i for i, d in enumerate(doc_ids)}
        by_tag = {}
        for doc, tags in per_doc.items():
            for tg in {x.lower() for x in tags if x}:
                by_tag.setdefault(tg, []).append(row_of[doc])
        tag_rows = [by_tag.get(q["predicates"][0].lower(), []) for q in queries]
        log(f"doc_label same-name tags: {sum(bool(x) for x in tag_rows)}/{len(queries)} queries have documents, "
            f"median {int(np.median([len(x) for x in tag_rows if x])) if any(tag_rows) else 0} documents")
    judge = Judge(ds, llm, texts, chunk_chars, audit_path=out / "format_repairs.jsonl")
    streams = Streams(queries, cand, near, len(doc_ids), cfg["per_round"], cfg["seed"],
                      sims_fn=lambda qi: emb @ Q[qi], tag_rows=tag_rows)
    meta = {"llm": llm.model, "labels": "teacher", "queries": "queries.json (all single-predicate queries)",
            "per_round": cfg["per_round"], "near_pool": cfg["near_pool"], "seed": cfg["seed"], "chunk_chars": chunk_chars,
            "sampling": "deterministic per-query stream: candidate set 60% / nearest neighbours 25% / uniform 15%, "
                        "refilled in that order; each block serves the queries with the fewest pairs",
            "seek": {**seek, "order": "doc_label same-name tag documents (cosine descending) -> whole corpus by cosine descending"}
            if policy else None,
            "parsing": "strict binary: only '0'/'1' accepted; up to 3 attempts with a repair message; failure raises"}
    n_chunks0, n_pairs0 = len(chunk_files(out)), sum(len(json.loads(f.read_text(encoding="utf-8"))["rows"]) for f in chunk_files(out))
    budgets, stage = open_stage(out, minutes, n_chunks0, n_pairs0, log)
    deadline = stage["deadline_epoch"] if stage else None
    calls0, hits0, t_run = llm.calls, llm.hits, now()
    reason, written = label_loop(queries, streams, doc_ids,
                                 lambda qis, pairs: label_pairs([queries[qi] for qi in qis], pairs, judge, llm, log, deadline=deadline),
                                 out, cfg["chunk_pairs"], meta, log, policy=policy, max_chunks=max_chunks, deadline=deadline)
    if stage:
        n_pairs1 = sum(len(json.loads(f.read_text(encoding="utf-8"))["rows"]) for f in chunk_files(out))
        stage["runs"].append({"started_epoch": t_run, "finished_epoch": now(), "reason": reason, "chunks_written": written,
                              "pairs_written": n_pairs1 - n_pairs0 if not stage["runs"] else n_pairs1 - stage["runs"][-1]["pairs_total"],
                              "pairs_total": n_pairs1, "llm_new_calls": llm.calls - calls0, "llm_cache_hits": llm.hits - hits0,
                              "judge_calls": judge.calls, "judge_unparsed": judge.unparsed, "format_repairs": judge.format_repairs})
        if reason in ("TIME_BUDGET", "ALL_DONE"):
            stage.update(finished_epoch=now(), reason=reason, elapsed_seconds=now() - stage["started_epoch"],
                         drain_seconds=max(0.0, now() - stage["deadline_epoch"]),
                         chunks_after=len(chunk_files(out)), pairs_after=n_pairs1,
                         new_pairs=n_pairs1 - stage["pairs_before"],
                         llm_new_calls=sum(r["llm_new_calls"] for r in stage["runs"]),
                         llm_cache_hits=sum(r["llm_cache_hits"] for r in stage["runs"]))
        write_json(out / "budgets.json", budgets, indent=1)
    log(f"LLM calls {llm.calls}, cache hits {llm.hits}, teacher verdicts {judge.calls}, unparsed {judge.unparsed}, "
        f"format repairs {judge.format_repairs}; {reason}")
    return reason


def snapshot(labels_dir, out, log=print):
    """Freeze the current data_single.json export and the query vectors into out/; later labeling does not change it.
    The block count is found by matching the cumulative per-block pair count to the exported pair count."""
    src, out = Path(labels_dir), Path(out)
    if out.exists():
        raise SystemExit(f"{out} already exists: snapshots are not overwritten; choose another name")
    d = json.loads((src / "data_single.json").read_text(encoding="utf-8"))
    total, k = 0, None
    for i, f in enumerate(chunk_files(src), 1):
        total += len(json.loads(f.read_text(encoding="utf-8"))["rows"])
        if total == len(d["rows"]):
            k = i
            break
    if k is None:
        raise SystemExit(f"{src}: cumulative per-block pair count does not match the exported {len(d['rows'])} pairs")
    z = np.load(src / "stream_inputs.npz", allow_pickle=False)
    if z["queries"].tolist() != [q["query"] for q in d["queries"]]:
        raise SystemExit("stream_inputs.npz query list does not match data_single.json")
    budgets = json.loads((src / "budgets.json").read_text(encoding="utf-8")) if (src / "budgets.json").exists() else []
    snap = {"source": str(src), "chunks": k, "pairs": len(d["rows"]), "frozen": time.strftime("%F %T"),
            "stages": [{k_: s.get(k_) for k_ in ("stage", "requested_minutes", "started_epoch", "deadline_epoch", "finished_epoch",
                                                  "reason", "pairs_before", "pairs_after", "llm_new_calls", "llm_cache_hits")}
                       for s in budgets]}
    write_json(out / "data_single.json", {**d, "meta": {**d["meta"], "snapshot": snap}})
    write_npz(out / "query_vectors.npz", queries=z["queries"], Q=z["Q"])
    log(f"Snapshot {out.name}: {k} blocks, {len(d['rows'])} pairs, {len(d['queries'])} queries, positive rate {d['meta']['pos_rate']:.1%}")
    return snap
