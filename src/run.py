"""Command-line entry point. One run (--run NAME) holds one catalog, index, query set and label stream under
<workspace>/runs/NAME/{catalog, index, query_gen, labels, snapshots/<name>, models/<name>, estimates/<name>}.
Relative paths in the config are resolved against the config file's directory.

  python run.py --config "config path" --run <name> embed                         # document embeddings (once per dataset)
  ... build-catalog [--reuse-phrases]                                           # Top2Vec phrases -> LLM dimensions -> catalog + glosses
  ... build-index                                                               # index.mode single | multi
  ... gen-queries                                                               # training queries (single predicate)
  ... label [--chunks N] [--minutes M]                                          # teacher labels; stop with `touch labels/STOP`, rerun to resume
  ... snapshot --name <name>                                                    # freeze labels + query vectors
  ... train --snapshot <name> --name <model> [--protocol kfold|split] [--k 5] [--device cuda]
  ... calibrate --model <model> [--device cuda]                                 # AND threshold (judge.and_judge = per_predicate_calibrated)
  ... estimate --model <model>|llm --name <eval> [--sets single,and,or] [--limit N]
  ... evaluate --name <eval>                                                    # q-error against evaluation.labels
"""
import argparse
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

import calibrate  # noqa: E402
import catalog as catalog_mod  # noqa: E402
import data  # noqa: E402
import embed  # noqa: E402
import estimate  # noqa: E402
import evaluate  # noqa: E402
import index_multi  # noqa: E402
import label  # noqa: E402
import querygen  # noqa: E402
import train  # noqa: E402
from index import Teacher, build, load_index, save_index  # noqa: E402
from judge import Judge, MLPJudge, make_query  # noqa: E402
from llm import LLM  # noqa: E402
from sample import AND_WEIGHTS  # noqa: E402
from selector import Selector  # noqa: E402

INDEX_MODES = ("single", "multi")
AND_JUDGES = ("joint", "per_predicate_calibrated")
DEFAULT_MINUTES = 600.0   # labeling time budget in minutes when --minutes is not given


def log_line(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_cfg(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_dir"] = path.parent
    data.check_dataset(cfg["dataset"])
    mode = cfg["index"].get("mode", "single")
    if mode not in INDEX_MODES:
        raise ValueError(f"index.mode must be one of {INDEX_MODES}, got {mode!r}")
    if mode == "multi" and not cfg["dataset"].get("multi"):
        raise ValueError("index.mode = multi requires dataset.multi")
    if and_weight_of(cfg) not in AND_WEIGHTS:
        raise ValueError(f"sample.and_weight must be one of {AND_WEIGHTS}")
    if and_judge_of(cfg) not in AND_JUDGES:
        raise ValueError(f"judge.and_judge must be one of {AND_JUDGES}")
    if (cfg["query_gen"].get("fulltext") or {}).get("enabled") and not cfg["dataset"].get("fulltext"):
        raise ValueError("query_gen.fulltext.enabled requires dataset.fulltext")
    return cfg


def and_weight_of(cfg):
    return cfg["sample"].get("and_weight", "query")


def and_judge_of(cfg):
    return cfg["judge"].get("and_judge", "joint")


def budget_of(cfg, n_docs):
    """Verdict budget per query = floor(n_docs * sample.budget_ratio)."""
    return int(n_docs * float(cfg["sample"]["budget_ratio"]))


def resolve(cfg, p):
    p = Path(p)
    return p if p.is_absolute() else cfg["_dir"] / p


def make_llm(cfg, override=None, cache=None, timeout=600):
    c = dict(cfg["llm"], **(override or {}))
    return LLM(c["base_url"], c["model"], cache or resolve(cfg, c["cache"]), c.get("workers", 16),
               timeout=timeout, extra_body=c.get("extra_body"))


def make_embedder(cfg):
    c = cfg["embedding"]
    return LLM(c["base_url"], c["model"], cache_path=None, workers=4)


def make_qvecs(cfg):
    return data.QueryVectors(resolve(cfg, cfg["embedding"]["query_vectors"]), make_embedder(cfg))


def run_dir(cfg, args):
    return resolve(cfg, cfg["workspace"]) / "runs" / args.run


def need(path, hint):
    if not Path(path).exists():
        raise SystemExit(f"missing {path}: {hint}")
    return path


def load_corpus(cfg):
    return data.load_corpus(resolve(cfg, cfg["data"]["corpus"]))


def load_emb(cfg, doc_ids):
    d = cfg["data"]
    return data.load_embeddings(need(resolve(cfg, d["embeddings"]), "run `embed` first"), doc_ids,
                                resolve(cfg, d["embedding_manifest"]) if d.get("embedding_manifest") else None)


def load_run_catalog(run):
    c = need(run / "catalog" / "semantic_catalog.json", "run build-catalog first")
    return data.load_catalog(c), data.load_json(run / "catalog" / "catalog_glosses.json")


def load_run_index(run, doc_ids):
    return load_index(need(run / "index" / "index.json", "run build-index first"), doc_ids)


def cmd_embed(args, cfg, run):
    doc_ids, texts = load_corpus(cfg)
    d, e = cfg["data"], cfg["embedding"]
    embed.run(doc_ids, texts, make_embedder(cfg), resolve(cfg, d["embeddings"]),
              resolve(cfg, d.get("embedding_manifest") or str(Path(d["embeddings"]).with_suffix(".manifest.json"))),
              int(e.get("max_tokens", 8192)), log=log_line)


def cmd_build_catalog(args, cfg, run):
    out = run / "catalog"
    if (out / "semantic_catalog.json").exists():
        raise SystemExit(f"{out} already has a catalog; use another --run")
    doc_ids, texts = load_corpus(cfg)
    emb = load_emb(cfg, doc_ids)
    llm = make_llm(cfg, cfg["catalog"].get("llm"), cache=out / "llm_cache.jsonl", timeout=1800)
    ccfg = dict(cfg["catalog"], reuse_phrases=bool(args.reuse_phrases), root_hint=cfg["dataset"]["root_hint"],
                dataset=cfg["dataset"])
    n_sample = catalog_mod.sample_count(len(doc_ids), ccfg["sample_ratio"], ccfg["min_sample_docs"])
    log_line(f"{cfg['dataset']['name']}: {len(doc_ids)} documents, {n_sample} sampled for the catalog; LLM {llm.model}; output {out}")
    catalog_mod.build_catalog(ccfg, doc_ids, texts, llm, emb=emb, embedder=make_embedder(cfg), out_dir=out, log=log_line)


def cmd_build_index(args, cfg, run):
    out = run / "index" / "index.json"
    if out.exists():
        raise SystemExit(f"{out} already exists; use another --run")
    doc_ids, texts = load_corpus(cfg)
    emb = load_emb(cfg, doc_ids)
    catalog, glosses = load_run_catalog(run)
    names = catalog.names
    vecs = make_embedder(cfg).embed([f"{n}: {glosses[n]}" if glosses.get(n) else n for n in names])
    name_emb = dict(zip(names, vecs))
    llm, b, mode = make_llm(cfg), cfg["index"], cfg["index"].get("mode", "single")
    t0 = time.time()
    extra = {}
    if mode == "single":
        teacher = Teacher(cfg["dataset"], llm, glosses, votes=int(b["votes"]), chunk_chars=cfg["judge"]["chunk_chars"])
        node_to_rows, stats = build(b, teacher, doc_ids, texts, emb, name_emb, catalog, log=log_line)
    else:
        mdir = run / "index" / "index_multi"
        teacher = index_multi.MultiTeacher(llm, glosses, cfg["dataset"]["multi"], chunk_chars=cfg["judge"]["chunk_chars"],
                                           audit_path=mdir / "format_repairs.jsonl")
        node_to_rows, stats, supervision, node_records = index_multi.build_sampled(
            b["multi"], teacher, doc_ids, texts, emb, name_emb, catalog, mdir / "classifiers", log=log_line)
        data.write_json(mdir / "teacher_assignments.json", supervision, indent=1)
        data.write_json(mdir / "nodes.json", node_records, indent=1)
        import numpy as np
        label.write_npz(mdir / "node_vectors.npz", names=np.array(names), V=np.stack([name_emb[n] for n in names]))
        extra = {"multi_config": dict(b["multi"]), "teacher_format_repairs": teacher.format_repairs}
    save_index(out, node_to_rows, doc_ids, provenance={
        "run": args.run, "dataset": cfg["dataset"]["name"], "mode": mode, "llm": llm.model,
        "index_config": {k: v for k, v in b.items() if k != "multi"}, "stats": stats, "seconds": round(time.time() - t0),
        "llm_new_calls": llm.calls, "llm_cache_hits": llm.hits, **extra})
    log_line(f"index ({mode}) written to {out}: {len(node_to_rows)} nodes, {sum(len(v) for v in node_to_rows.values())} memberships; "
             f"{llm.calls} LLM calls, {llm.hits} cache hits, {time.time() - t0:.0f}s")


def cmd_gen_queries(args, cfg, run):
    out = run / "query_gen"
    if (out / "queries.json").exists():
        raise SystemExit(f"{out} already has queries; use another --run")
    doc_ids, texts = load_corpus(cfg)
    catalog, glosses = load_run_catalog(run)
    index_rows = load_run_index(run, doc_ids)
    llm = make_llm(cfg, timeout=900)
    querygen.run(cfg["query_gen"], cfg["dataset"], catalog, glosses, index_rows, doc_ids, texts, llm, out, log=log_line)
    log_line(f"{llm.calls} LLM calls, {llm.hits} cache hits")


def merge_per_doc(per_doc, fulltext_records):
    """Per-document labels for positive seeking = doc_label tiers + full-text labels (case-insensitive, order kept)."""
    out = {d: list(v) for d, v in per_doc.items()}
    for rec in fulltext_records:
        names = {str(s).lower(): s for s in out.get(rec["doc_id"], []) if s}
        for lab in rec["labels"]:
            names.setdefault(lab.lower(), lab)
        out[rec["doc_id"]] = list(names.values())
    return out


def cmd_label(args, cfg, run):
    doc_ids, texts = load_corpus(cfg)
    emb = load_emb(cfg, doc_ids)
    catalog, _ = load_run_catalog(run)
    index_rows = load_run_index(run, doc_ids)
    queries = data.load_json(need(run / "query_gen" / "queries.json", "run gen-queries first"))
    per_doc = data.load_json(run / "query_gen" / "doc_label.json")["meta"]["extra"]["per_doc"]
    if (run / "query_gen" / "fulltext.json").exists():
        per_doc = merge_per_doc(per_doc, data.load_json(run / "query_gen" / "fulltext.json")["records"])
    llm = make_llm(cfg)
    minutes = args.minutes if args.minutes is not None else DEFAULT_MINUTES
    log_line(f"LLM {llm.model}; {len(index_rows)} index nodes; output {run / 'labels'}; time budget {minutes} min")
    label.run(cfg["label"], cfg["dataset"], queries, catalog, index_rows, doc_ids, texts, emb, make_qvecs(cfg), llm, per_doc,
              run / "labels", cfg["judge"]["chunk_chars"], cfg["select"], max_chunks=args.chunks, minutes=minutes, log=log_line)


def cmd_snapshot(args, cfg, run):
    need(run / "labels" / "data_single.json", "run label first")
    label.snapshot(run / "labels", run / "snapshots" / args.name, log=log_line)


def cmd_train(args, cfg, run):
    t = cfg["train"]
    doc_ids, _ = load_corpus(cfg)
    emb = load_emb(cfg, doc_ids)
    snap = need(run / "snapshots" / args.snapshot, "run snapshot first")
    train.run(t, snap, run / "models" / args.name, doc_ids, emb, protocol=args.protocol or t["protocol"],
              k=args.k or t["k"], device=args.device or t.get("device", "cpu"), log=log_line)


def cmd_calibrate(args, cfg, run):
    doc_ids, _ = load_corpus(cfg)
    emb = load_emb(cfg, doc_ids)
    mdir = need(run / "models" / args.model, "run train first")
    snap = need(run / "snapshots" / data.load_json(mdir / "meta.json")["snapshot"], "snapshot used for training is missing")
    calibrate.fit(mdir, snap, doc_ids, emb, device=args.device or cfg["train"].get("device", "cpu"), log=log_line)


def cmd_estimate(args, cfg, run):
    doc_ids, texts = load_corpus(cfg)
    emb = load_emb(cfg, doc_ids)
    catalog, _ = load_run_catalog(run)
    index_rows = load_run_index(run, doc_ids)
    llm, qvecs, s = make_llm(cfg), make_qvecs(cfg), cfg["sample"]
    budget = budget_of(cfg, len(doc_ids))
    sets = (args.sets or ",".join(cfg["evaluation"]["queries"])).split(",")
    queries_by_set = {k: data.load_queries(resolve(cfg, cfg["evaluation"]["queries"][k]))[:args.limit or None] for k in sets}
    all_q = [q for qs in queries_by_set.values() for q in qs]
    qvecs.matrix([q["query"] for q in all_q] + [make_query([p], "single") for q in all_q for p in q["predicates"]])
    out = run / "estimates" / args.name
    and_judge, and_thr = and_judge_of(cfg), {}
    if args.model == "llm":
        judges = {".": Judge(cfg["dataset"], llm, texts, cfg["judge"]["chunk_chars"], audit_path=out / "format_repairs.jsonl")}
        and_judge = "llm"
    else:
        mdir = need(run / "models" / args.model, "run train first")
        top = data.load_json(mdir / "meta.json")
        names = [f"fold{k}" for k in range(top["k"])] if top["protocol"] == "kfold" else ["."]
        if and_judge == "per_predicate_calibrated":
            need(mdir / "and_calibration.json", "run calibrate first")
            and_thr = calibrate.load_thresholds(mdir, names)
        judges = {n: MLPJudge(mdir / n if n != "." else mdir, emb, qvecs, and_threshold=and_thr.get(n)) for n in names}
    selector = Selector(catalog, llm=llm, votes=cfg["select"]["votes"], temperature=cfg["select"]["temperature"], log=log_line)
    log_line(f"{cfg['dataset']['name']}: {len(doc_ids)} documents, budget {budget}/query; {len(index_rows)} index nodes; "
             f"selector {llm.model}; judge {args.model} ({len(judges)} models); and_weight {and_weight_of(cfg)}, and_judge {and_judge}; "
             f"sets {sets}; output {out}")
    meta = {"run": args.run, "dataset": cfg["dataset"]["name"], "index_mode": cfg["index"].get("mode", "single"), "llm": llm.model,
            "judge": args.model, "budget": budget, "budget_ratio": float(s["budget_ratio"]), "n_docs": len(doc_ids),
            "seed": int(s["seed"]), "limit": args.limit, "sets": sets, "and_judge": and_judge, "and_thresholds": and_thr or None,
            "select": dict(cfg["select"]), "chunk_chars": cfg["judge"]["chunk_chars"]}
    result = estimate.run(queries_by_set, selector, index_rows, emb, qvecs, doc_ids, judges, llm, budget,
                          random.Random(int(s["seed"])), out, meta, catalog, log=log_line, and_weight=and_weight_of(cfg))
    log_line(f"results in {result}; {llm.calls} LLM calls, {llm.hits} cache hits")


def cmd_evaluate(args, cfg, run):
    est = run / "estimates" / args.name
    meta = data.load_json(need(est / "meta.json", "run estimate first"))
    doc_ids, _ = load_corpus(cfg)
    e = cfg["evaluation"]
    evaluate.report(est / meta["result"], list(meta["sets"]), {k: resolve(cfg, p) for k, p in e["queries"].items()},
                    need(resolve(cfg, e["labels"]), "evaluation.labels is required"), doc_ids,
                    label=f"{args.run}/{args.name}", out_dir=est)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed")
    p = sub.add_parser("build-catalog")
    p.add_argument("--reuse-phrases", action="store_true", help="reuse catalog/phrases.json from a previous attempt")
    sub.add_parser("build-index")
    sub.add_parser("gen-queries")
    p = sub.add_parser("label")
    p.add_argument("--chunks", type=int, default=0, help="write at most N more chunks (0 = until STOP / budget / exhausted)")
    p.add_argument("--minutes", type=float, default=None, help="time budget in minutes (default 600; 0 = unlimited)")
    p = sub.add_parser("snapshot")
    p.add_argument("--name", required=True)
    p = sub.add_parser("train")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--protocol", choices=["kfold", "split"])
    p.add_argument("--k", type=int)
    p.add_argument("--device")
    p = sub.add_parser("calibrate")
    p.add_argument("--model", required=True)
    p.add_argument("--device")
    p = sub.add_parser("estimate")
    p.add_argument("--model", required=True, help="model name from train, or `llm` for direct LLM verdicts")
    p.add_argument("--name", required=True)
    p.add_argument("--sets", default="", help="default: all sets in evaluation.queries")
    p.add_argument("--limit", type=int, default=0, help="only the first N queries per set")
    p = sub.add_parser("evaluate")
    p.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    run = run_dir(cfg, args)
    {"embed": cmd_embed, "build-catalog": cmd_build_catalog, "build-index": cmd_build_index, "gen-queries": cmd_gen_queries,
     "label": cmd_label, "snapshot": cmd_snapshot, "train": cmd_train, "calibrate": cmd_calibrate,
     "estimate": cmd_estimate, "evaluate": cmd_evaluate}[args.cmd](args, cfg, run)


if __name__ == "__main__":
    main()
