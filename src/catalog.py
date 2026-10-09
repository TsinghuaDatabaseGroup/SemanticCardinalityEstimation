"""Automatic catalog (taxonomy tree) construction.
Pipeline: phrase extraction (contextualized Top2Vec) -> LLM dimension identification (heap loop over nodes)
-> top-K children by document frequency -> pruning -> per-node glosses -> self-check.
Outputs semantic_catalog.json and catalog_glosses.json (consumed by index.py / selector.py) plus
catalog_build.json (config snapshot, per-dimension phrases and documents, sibling-similarity report, self-check).
"""
import heapq
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from data import Catalog, normalize
from judge import split_chunks

RESERVED = {"root", "others", "other", "miscellaneous", "other topics", "misc"}
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 &'-]{1,39}")
DEFAULT_SYSTEM_PROMPT = ("\nYou are a helpful assistant. You are provided with a question. "
                         "Your task is to answer the question succinctly.\n")

# ---------------------------------------------------------------- Prompts
IDENTIFY_PROMPT = """You are a semantic analysis expert. Given the following set of key phrases,
    identify new semantic dimensions (high-level concepts) that group related phrases together.
    Avoid repeating existing dimensions.

    Existing dimensions: {existing}
    Phrases (numbered):
{phrases}

    Please return JSON of the form:
    {{
      "DimensionName1": [1, 5, 9],
      "DimensionName2": [2, 3]
    }}
    Keep the number of dimensions under {max_dims}.
    Each dimension name is a short Title Case category name of 1-4 common-noun words (e.g. "Military Conflicts"),
    not a phrase or sentence, with no proper names; it must not repeat or paraphrase an existing dimension or "{parent}".
    Assign every phrase number to exactly one dimension."""

OMITTED_PROMPT = """You previously grouped key phrases into these dimensions: {names}.
    The following phrases were left out. Assign each of them to one of the dimensions above, or to a new
    dimension if none fits (same naming rules: short Title Case category name, 1-4 common-noun words).
    Phrases (numbered):
{phrases}
    Return only JSON of the form {{"DimensionName": [numbers], ...}}."""

SAME_PROMPT = """You are an expert in semantic similarity.

I will give you two short category names. Your task is to determine whether they refer to the **same or highly similar concept**.

Rules:
- Consider semantic equivalence, not exact string match.
- If the meanings overlap strongly or are nearly the same, answer "YES".
- If they refer to different concepts, answer "NO".

Now judge:

Name A: "{a}"
Name B: "{b}"

Answer **only YES or NO**."""

GLOSS_PROMPT = """You are writing {owner} classification catalog. Under the category "{parent}" there are the
following subcategories; each is followed by key phrases of the documents that belong to it.
{lines}
For each subcategory write one sentence (under 30 words) saying which documents belong to it; when two sibling
subcategories could be confused, say where the boundary lies. Do not start the sentence with the subcategory name.
Return JSON of the form: {{"SubcategoryName": "sentence", ...}} using the subcategory names exactly as written."""

OTHERS_GLOSS = "Documents whose primary subject fits none of the other top-level categories."


# ---------------------------------------------------------------- LLM calls and parsing
def parse_json(text):
    s = re.sub(r"^```(json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in reply")
    obj, _ = json.JSONDecoder().raw_decode(s[min(starts):])
    return obj


def ask_json(llm, prompt, check, max_tokens, tries=3, log=print):
    """Bounded retry: when parsing or `check` raises ValueError, the reason is appended to the prompt and the
    question is asked again; on an LLM error the failure is logged and None is returned."""
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    for attempt in range(tries):
        try:
            raw = llm.chat(messages, temperature=0.0, max_tokens=max_tokens)[0]
            return check(parse_json(raw))
        except RuntimeError as err:
            log(f"  LLM call failed, giving up: {err}")
            return None
        except (ValueError, TypeError) as err:
            log(f"  reply rejected (attempt {attempt + 1}): {str(err)[:160]}")
            messages = messages[:1] + [{"role": "user", "content":
                                       prompt + f"\n\nYour previous answer was rejected: {err}. "
                                       "Return only the JSON."}]
    return None


def clean_name(raw):
    s = re.sub(r"\(.*?\)", "", str(raw)).replace(",", " and").replace("/", " and")
    s = re.sub(r"[^A-Za-z0-9 &'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or not NAME_RE.fullmatch(s) or len(s.split()) > 5 or s.lower() in RESERVED:
        return None
    return s


def make_check_dims(n_phrases, existing, parent, min_dims=2, min_share=0.2):
    """Validate {name: [phrase numbers]}: hard failures raise ValueError; soft problems are fixed in place
    and recorded in notes."""
    def check(obj):
        if not isinstance(obj, dict):
            raise ValueError("not a JSON object")
        dims, notes = {}, []
        for raw_name, nums in obj.items():
            name = clean_name(raw_name)
            if name is None:
                notes.append(f"dropped name {raw_name!r}")
                continue
            if name.lower() == parent.lower():
                notes.append(f"dropped {name!r}: same as parent")
                continue
            if name.lower() in existing:
                renamed = clean_name(f"{parent} {name}") if parent.lower() != "root" else None
                if renamed is None or renamed.lower() in existing:
                    notes.append(f"dropped {name!r}: duplicates an existing dimension")
                    continue
                notes.append(f"renamed {name!r} -> {renamed!r}")
                name = renamed
            if not isinstance(nums, list):
                nums = [nums]
            good = []
            for x in nums:
                try:
                    i = int(x)
                except (TypeError, ValueError):
                    continue
                if 1 <= i <= n_phrases and i not in good:
                    good.append(i)
            if not good:
                notes.append(f"dropped {name!r}: no valid phrase numbers")
                continue
            dims.setdefault(name, [])
            dims[name].extend(i for i in good if i not in dims[name])
        if len(dims) < min_dims:
            raise ValueError(f"only {len(dims)} valid dimensions (need {min_dims}); notes={notes[:4]}")
        assigned = {i for v in dims.values() for i in v}
        if len(assigned) < min_share * n_phrases:
            raise ValueError(f"only {len(assigned)}/{n_phrases} phrases were assigned")
        return dims, notes
    return check


def check_gloss_factory(children):
    import difflib

    def check(obj):
        if not isinstance(obj, dict):
            raise ValueError("not a JSON object")
        out = {}
        for k, v in obj.items():
            if not isinstance(v, str) or not v.strip():
                continue
            hit = k if k in children else (difflib.get_close_matches(str(k), children, n=1, cutoff=0.8) or [None])[0]
            if hit:
                out[hit] = v.strip()
        if len(out) < max(1, len(children) // 2):
            raise ValueError(f"only {len(out)}/{len(children)} subcategories described")
        return out
    return check


# ---------------------------------------------------------------- Phrase extraction: contextualized Top2Vec
def sample_count(n, ratio, min_docs):
    """Number of documents to sample: a `ratio` share of n but at least `min_docs`; when fewer than
    ratio * n documents would be left over, all n are used."""
    base = min(n, max(1, int(n * ratio), int(min_docs)))
    return n if n - base < ratio * n else base


def phrases_top2vec(doc_ids, texts, cfg, log=print):
    """Sample -> chunk -> contextual Top2Vec -> for each document take the top N topic words of its main topic.
    Returns (doc_phrase_map, doc_topic, topic_phrases, stats)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import top2vec.top2vec as t2v_mod
    from top2vec import Top2Vec
    t = cfg["top2vec"]
    if t.get("model_path"):
        t2v_mod.contextual_top2vec_model_paths[t["embedding_model"]] = t["model_path"]
    rng = random.Random(cfg["seed"])
    n = sample_count(len(doc_ids), cfg["sample_ratio"], cfg["min_sample_docs"])
    sample = sorted(rng.sample(range(len(doc_ids)), n))
    chunks, owner = [], []
    for i in sample:
        for c in split_chunks(texts[i], t["chunk_chars"]):
            chunks.append(c)
            owner.append(i)
    owner = np.asarray(owner)
    log(f"Top2Vec: sampled {n} docs, split into {len(chunks)} chunks (each <= {t['chunk_chars']} chars)")
    t0 = time.time()
    model = Top2Vec(documents=chunks, contextual_top2vec=True,
                    embedding_model=t["embedding_model"], embedding_model_path=t.get("model_path"),
                    min_count=t["min_count"], ngram_vocab=True,
                    c_top2vec_smoothing_window=t.get("smoothing_window", 5),
                    topic_merge_delta=t.get("topic_merge_delta", 0.1),
                    hdbscan_args={"min_cluster_size": t["min_cluster_size"], "min_samples": t["min_samples"]},
                    verbose=False)
    n_topics = model.get_num_topics()
    target = t.get("num_topics")
    if target and n_topics > target:                  # merge topics hierarchically down to the target count
        model.hierarchical_topic_reduction(target)
        words, _, nums = model.get_topics(target, reduced=True)
        dist = np.asarray(model.doc_topic_distribution_reduced)
        log(f"Top2Vec: {n_topics} topics, hierarchically reduced to {target}, {time.time() - t0:.0f}s")
    else:
        words, _, nums = model.get_topics(n_topics)
        dist = np.asarray(model.get_document_topic_distribution())
        log(f"Top2Vec: {n_topics} topics, {time.time() - t0:.0f}s")
    topic_phrases = {}
    for w, k in zip(words, nums):
        out = []
        for x in w:
            p = str(x).replace("_", " ").strip().lower()
            if p and len(p.split()) <= cfg["max_phrase_len"] and re.fullmatch(r"[a-z0-9][a-z0-9 '-]*", p) and p not in out:
                out.append(p)
            if len(out) >= cfg["phrases_per_doc"]:
                break
        topic_phrases[int(k)] = out
    doc_topic, doc_phrase_map = {}, {}                                # dist: chunk x topic
    for i in sample:
        rows = np.where(owner == i)[0]
        k = int(dist[rows].sum(axis=0).argmax())
        doc_topic[doc_ids[i]] = k
        doc_phrase_map[doc_ids[i]] = list(topic_phrases.get(k, []))
    sizes = Counter(doc_topic.values())
    stats = {"sampled_docs": n, "chunks": len(chunks), "topics": n_topics,
             "docs_per_topic": {"min": min(sizes.values()), "median": float(np.median(list(sizes.values()))),
                                "max": max(sizes.values())}, "seconds": round(time.time() - t0)}
    return doc_phrase_map, doc_topic, topic_phrases, stats


def clean_phrases(doc_phrase_map, topic_phrases, letters_only, log=print):
    """Optional: keep only all-letter phrases (drops topic words containing digits such as 'season 2')."""
    if not letters_only:
        return doc_phrase_map, topic_phrases
    ok = lambda p: re.fullmatch(r"[a-z][a-z '-]*", p) is not None  # noqa: E731
    dropped = sorted({p for ps in topic_phrases.values() for p in ps if not ok(p)})
    if dropped:
        log(f"dropped {len(dropped)} phrases containing digits: {dropped[:10]}")
    return ({d: [p for p in ps if ok(p)] for d, ps in doc_phrase_map.items()},
            {k: [p for p in ps if ok(p)] for k, ps in topic_phrases.items()})


def drop_stop_phrases(doc_phrase_map, topic_phrases, share, log=print):
    """Phrases appearing in more than a `share` fraction of the topics are treated as generic and removed
    from every document."""
    n_topics = len(topic_phrases)
    if n_topics < 10:
        return doc_phrase_map, []
    cnt = Counter(p for ps in topic_phrases.values() for p in set(ps))
    stop = sorted(p for p, c in cnt.items() if c > share * n_topics)
    if stop:
        log(f"{len(stop)} stop phrases (appear in >{share:.0%} of topics): {stop[:12]}")
        doc_phrase_map = {d: [p for p in ps if p not in stop] for d, ps in doc_phrase_map.items()}
    return doc_phrase_map, stop


# ---------------------------------------------------------------- Dimension identification, heap loop
def ordered_phrases(doc_ids, doc_phrase_map):
    freq = Counter(p for d in doc_ids for p in set(doc_phrase_map[d]))
    return [p for p, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]


def numbered(phrases):
    return "\n".join(f"    {i + 1}. {p}" for i, p in enumerate(phrases))


def identify(llm, phrases, existing, parent, max_dims, cfg, log):
    """Returns ({dimension name: [phrases]}, notes) or None. Includes one follow-up query for omitted phrases."""
    prompt = IDENTIFY_PROMPT.format(existing=sorted(existing.values()), phrases=numbered(phrases),
                                    max_dims=max_dims, parent=parent)
    if parent == "Root" and cfg.get("root_hint"):
        prompt += "\n    " + cfg["root_hint"]
    res = ask_json(llm, prompt, make_check_dims(len(phrases), existing, parent), cfg["max_tokens_identify"],
                   cfg["tries"], log)
    if res is None:
        return None
    dims, notes = res
    assigned = {i for v in dims.values() for i in v}
    left = [i for i in range(1, len(phrases) + 1) if i not in assigned]
    if left and len(left) > (1 - cfg["min_assigned_share"]) * len(phrases):
        log(f"  {len(left)}/{len(phrases)} phrases omitted, asking once more")
        sub = [phrases[i - 1] for i in left]
        taken = dict(existing); taken.update({n.lower(): n for n in dims})
        res2 = ask_json(llm, OMITTED_PROMPT.format(names=list(dims), phrases=numbered(sub)),
                        make_check_dims(len(sub), {k: v for k, v in taken.items() if k not in {n.lower() for n in dims}},
                                        parent, min_dims=1, min_share=0.0),
                        cfg["max_tokens_identify"], cfg["tries"], log)
        if res2:
            extra, notes2 = res2
            notes += notes2
            for name, nums in extra.items():
                key = next((n for n in dims if n.lower() == name.lower()), name)
                dims.setdefault(key, [])
                dims[key].extend(left[i - 1] for i in nums if left[i - 1] not in dims[key])
    return {name: [phrases[i - 1] for i in nums] for name, nums in dims.items()}, notes


def attach_leftovers(dims, phrases, embedder, log):
    """Phrases left unassigned after both LLM rounds are attached to the nearest dimension by embedding
    (dimension vector = mean of its phrase vectors)."""
    assigned = {p for v in dims.values() for p in v}
    left = [p for p in phrases if p not in assigned]
    if not left or embedder is None or not dims:
        return len(left)
    names = list(dims)
    vecs = embedder.embed(left + [p for n in names for p in dims[n]])
    left_v, rest = vecs[:len(left)], vecs[len(left):]
    cents, k = [], 0
    for n in names:
        cents.append(normalize(rest[k:k + len(dims[n])].mean(axis=0)))
        k += len(dims[n])
    best = (left_v @ np.stack(cents).T).argmax(axis=1)
    for p, j in zip(left, best):
        dims[names[j]].append(p)
    log(f"  {len(left)} leftover phrases attached to the nearest dimension by embedding")
    return len(left)


def partition_docs(node_docs, doc_phrase_map, dims):
    """Each document goes to the dimension that holds most of its phrases (ties go to the earlier dimension).
    Returns {dimension: [doc_id]}."""
    owner = {}
    for name, ps in dims.items():
        for p in ps:
            owner.setdefault(p, name)
    order = {n: i for i, n in enumerate(dims)}
    docs = defaultdict(list)
    for d in node_docs:
        votes = Counter(owner[p] for p in doc_phrase_map[d] if p in owner)
        if votes:
            docs[max(votes, key=lambda n: (votes[n], -order[n]))].append(d)
    return docs


def sibling_report(llm, selected, dim_docs, emb, row_of, merge_cos, log):
    """For sibling pairs whose document-centroid cosine >= merge_cos, ask the LLM once whether the names are
    synonymous. Report only; nothing is merged."""
    cents = {}
    for name, _ in selected:
        rows = [row_of[d] for d in dim_docs[name] if d in row_of]
        if rows:
            cents[name] = normalize(emb[rows].mean(axis=0))
    names = list(cents)
    report = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cos = float(cents[names[i]] @ cents[names[j]])
            if cos >= merge_cos:
                raw = llm.chat([{"role": "user", "content": SAME_PROMPT.format(a=names[i], b=names[j])}],
                               temperature=0.0, max_tokens=2)[0]
                same = raw.strip().upper().startswith("YES")
                report.append({"a": names[i], "b": names[j], "centroid_cos": round(cos, 3), "llm_same": same})
                log(f"  sibling candidate: {names[i]!r} vs {names[j]!r} cosine {cos:.3f}, LLM says {'YES' if same else 'NO'}")
    return report


def build_tree(doc_phrase_map, llm, cfg, emb=None, row_of=None, embedder=None, log=print):
    all_docs = sorted(doc_phrase_map)
    n_s = len(all_docs)
    root = {"name": "Root", "children": []}
    existing = {}                                   # lowercase name -> original name, shared across the tree
    dim_phrases, dim_docs, dropped, notes_all, sib = {}, {}, [], [], []
    heap, tie = [], 0
    root_phrases = ordered_phrases(all_docs, doc_phrase_map)
    if len(root_phrases) > cfg["max_root_phrases"]:                   # cap the prompt size
        log(f"root has {len(root_phrases)} phrases, keeping the top {cfg['max_root_phrases']} by document frequency")
        root_phrases = root_phrases[:cfg["max_root_phrases"]]
    heapq.heappush(heap, (-n_s, tie, root_phrases, all_docs, root, 0))
    while heap:
        neg_f, _, phrases, node_docs, node, depth = heapq.heappop(heap)
        if depth >= cfg["H"] or len(phrases) < cfg["min_phrases"] or len(node_docs) < cfg["min_docs"]:
            continue
        k_max = cfg["K_root"] if depth == 0 else cfg["K"]
        min_freq = int((cfg["min_freq_root"] if depth == 0 else cfg["min_freq"]) * n_s)
        log(f"== node '{node['name']}' depth {depth}, {len(phrases)} phrases, {len(node_docs)} docs")
        res = identify(llm, phrases, existing, node["name"], k_max if depth else cfg["max_new_dims"], cfg, log)
        if res is None:
            log("  dimension identification failed, node not expanded")
            continue
        dims, notes = res
        notes_all += [f"{node['name']}: {x}" for x in notes]
        attach_leftovers(dims, phrases, embedder, log)
        docs_of = partition_docs(node_docs, doc_phrase_map, dims)
        cand = sorted(((n, len(docs_of.get(n, []))) for n in dims), key=lambda kv: -kv[1])
        selected = []
        for name, f in cand:
            if f < max(1, min_freq):
                dropped.append({"node": node["name"], "dimension": name, "freq": f, "why": f"freq < {min_freq}",
                                "phrases": dims[name][:8]})
            elif len(selected) >= k_max:
                dropped.append({"node": node["name"], "dimension": name, "freq": f, "why": f"beyond top-{k_max}",
                                "phrases": dims[name][:8]})
            else:
                selected.append((name, f))
        for name, f in selected:
            dim_docs[name] = docs_of[name]
            dim_phrases[name] = dims[name]
        if emb is not None and len(selected) > 1:
            sib += sibling_report(llm, selected, dim_docs, emb, row_of, cfg["merge_cos"], log)
        for name, f in selected:
            existing[name.lower()] = name
            child = {"name": name, "freq": f, "children": []}
            node["children"].append(child)
            tie += 1
            heapq.heappush(heap, (-f, tie, ordered_phrases(docs_of[name], doc_phrase_map), docs_of[name],
                                  child, depth + 1))
        unassigned = len(node_docs) - sum(len(v) for v in docs_of.values())
        log("  -> " + ", ".join(f"{n}:{f}" for n, f in selected)
            + f"; unassigned {unassigned} docs; dropped {len(cand) - len(selected)} dimensions")
    return root, dim_phrases, dim_docs, dropped, notes_all, sib


# ---------------------------------------------------------------- Pruning, glosses, output format
def prune(node, dropped, is_root=True):
    for c in node["children"]:
        prune(c, dropped, False)
    if not is_root and len(node["children"]) == 1 and not node["children"][0]["children"]:
        dropped.append({"node": node["name"], "dimension": node["children"][0]["name"],
                        "freq": node["children"][0].get("freq"), "why": "pruned single leaf child"})
        node["children"] = []


def all_nodes(node):
    out = [node]
    for c in node["children"]:
        out += all_nodes(c)
    return out


def write_glosses(root, dim_phrases, llm, cfg, log=print):
    parents = [n for n in all_nodes(root) if n["children"]]
    fallback = []
    ds = cfg["dataset"]

    def one(parent):
        kids = [c["name"] for c in parent["children"]]
        lines = "\n".join(f"- {k}: {', '.join(dim_phrases.get(k, [])[:12])}" for k in kids)
        pname = ds["whole_collection"] if parent["name"] == "Root" else parent["name"]
        out = ask_json(llm, GLOSS_PROMPT.format(owner=ds["catalog_owner"], parent=pname, lines=lines),
                       check_gloss_factory(kids), 1024, cfg["tries"], log) or {}
        for k in kids:
            if k not in out:
                out[k] = "Documents about " + ", ".join(dim_phrases.get(k, ["this subject"])[:4]) + "."
                fallback.append(k)
        return out
    glosses = {}
    for res in llm.map_groups([[p] for p in parents], one):
        glosses.update(res[0])
    return glosses, fallback


def to_catalog(node):
    out = {"category": node["name"]}
    if node["children"]:
        out["subcategories"] = [to_catalog(c) for c in node["children"]]
    return out


def self_check(catalog_json, glosses, cfg, log=print):
    cat = Catalog(catalog_json)                       # raises on duplicate names: final check
    names = [n for n in cat.names if n != cat.root_name]
    depth = {cat.root_name: 0}
    for n in names:
        depth[n] = depth[cat.parent[n]] + 1
    warns = []
    if max(depth.values()) > cfg["H"]:
        warns.append(f"depth {max(depth.values())} > H={cfg['H']}")
    if not 10 <= len(names) <= 80:
        warns.append(f"non-root node count {len(names)} outside [10, 80]")
    for n in names:
        if not glosses.get(n):
            warns.append(f"node {n!r} has no gloss")
        if not NAME_RE.fullmatch(n):
            warns.append(f"invalid node name: {n!r}")
    for n, kids in cat.children.items():
        if len(kids) == 1:
            warns.append(f"node {n!r} has a single child")
    for w in warns:
        log(f"  self-check warning: {w}")
    return {"nodes": len(names), "depth": max(depth.values()),
            "per_level": dict(Counter(depth[n] for n in names)), "warnings": warns}


# ---------------------------------------------------------------- Main pipeline
def build_catalog(cfg, doc_ids, texts, llm, emb=None, embedder=None, out_dir=None, log=print,
                  phrase_fn=None):
    """phrase_fn may be injected (used by tests): returns (doc_phrase_map, doc_topic, topic_phrases, stats)."""
    t0 = time.time()
    cache = Path(out_dir) / "phrases.json" if out_dir else None
    if cache and cfg.get("reuse_phrases") and cache.exists():       # reuse cached Top2Vec output
        c = json.loads(cache.read_text(encoding="utf-8"))
        doc_phrase_map, doc_topic = c["doc_phrase_map"], c["doc_topic"]
        topic_phrases = {int(k): v for k, v in c["topic_phrases"].items()}
        t2v_stats = c["stats"]
        log(f"reusing {cache}: {len(doc_phrase_map)} docs, {len(topic_phrases)} topics")
    else:
        doc_phrase_map, doc_topic, topic_phrases, t2v_stats = (phrase_fn or phrases_top2vec)(doc_ids, texts, cfg, log)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"doc_phrase_map": doc_phrase_map, "doc_topic": doc_topic,
                                         "topic_phrases": topic_phrases, "stats": t2v_stats},
                                        ensure_ascii=False), encoding="utf-8")
    doc_phrase_map, topic_phrases = clean_phrases(doc_phrase_map, topic_phrases,
                                                  cfg.get("letters_only_phrases", False), log)
    doc_phrase_map, stop = drop_stop_phrases(doc_phrase_map, topic_phrases, cfg["stop_phrase_share"], log)
    row_of = {d: i for i, d in enumerate(doc_ids)}
    calls0 = llm.calls
    root, dim_phrases, dim_docs, dropped, notes, sib = build_tree(
        doc_phrase_map, llm, cfg, emb, row_of, embedder, log)
    prune(root, dropped)
    keep = {n["name"] for n in all_nodes(root)}
    dim_phrases = {k: v for k, v in dim_phrases.items() if k in keep}
    dim_docs = {k: v for k, v in dim_docs.items() if k in keep}
    glosses, fallback = write_glosses(root, dim_phrases, llm, cfg, log)
    catalog_json = to_catalog(root)
    if cfg.get("add_others", True) and "Others" not in keep:
        catalog_json.setdefault("subcategories", []).append({"category": "Others"})
        glosses["Others"] = OTHERS_GLOSS
    check = self_check(catalog_json, glosses, cfg, log)
    meta = {"config": {k: v for k, v in cfg.items() if k != "dataset"}, "dataset": cfg["dataset"]["name"],
            "top2vec": t2v_stats, "stop_phrases": stop, "dimension_phrases": dim_phrases,
            "dimension_docs": dim_docs, "doc_topic": doc_topic, "topic_phrases": topic_phrases,
            "dropped": dropped, "notes": notes, "sibling_report": sib, "gloss_fallback": fallback,
            "self_check": check, "llm_calls": llm.calls - calls0, "seconds": round(time.time() - t0)}
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "semantic_catalog.json").write_text(json.dumps(catalog_json, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
        (out_dir / "catalog_glosses.json").write_text(json.dumps(glosses, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
        (out_dir / "catalog_build.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
        log(f"catalog written to {out_dir}: {check['nodes']} non-root nodes, depth {check['depth']}, "
            f"{meta['llm_calls']} LLM calls, {len(fallback)} fallback glosses, {meta['seconds']}s")
    return catalog_json, glosses, meta
