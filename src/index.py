"""Semantic index: offline construction plus load/save.
Construction walks the catalog top-down. At every node with children:
  1. a random sample of the node's documents is classified by the teacher LLM (a child name or NONE);
     the root sample rate is multiplied by root_boost;
  2. every child needs at least child_floor teacher-confirmed seeds; otherwise the floor_pool documents
     nearest to the child-name embedding are verified as well;
  3. every child gets a prototype = mean of its seed embeddings (the name embedding when it has no seeds);
     a "stay in parent" column is added when there are >= 3 NONE seeds;
  4. teacher-decided documents follow the teacher; the rest go to the most similar prototype by cosine,
     or nowhere when below tau (root level: Others; deeper levels: stay in the parent);
  5. the parent keeps all its members and each child is processed recursively; the root holds the whole corpus.
The teacher reads the whole document; over-long documents are split into chunks that are asked separately,
and the majority answer wins (ties go to the first chunk).
File format: {"format_version": 1, "node_to_docs": {node: [doc_id, ...]}, "provenance": {...}};
in memory the index maps node -> list of row indices.
"""
import difflib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from data import normalize
from judge import split_chunks

FORMAT_VERSION = 1


# ---- Load / save ----------------------------------------------------------------------
def load_index(path, doc_ids):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("format_version") != FORMAT_VERSION or "node_to_docs" not in raw:
        raise ValueError(f"{path}: index format_version {raw.get('format_version')}, only {FORMAT_VERSION} (node_to_docs) is supported")
    row_of = {d: i for i, d in enumerate(doc_ids)}
    out = {}
    for node, docs in raw["node_to_docs"].items():
        missing = [d for d in docs if d not in row_of]
        if missing:
            raise ValueError(f"index node {node!r} contains doc_ids missing from the corpus: {missing[:3]}")
        out[node] = [row_of[d] for d in docs]
    return out


def save_index(path, node_to_rows, doc_ids, provenance=None):
    path = Path(path)
    for node, rows in node_to_rows.items():
        if len(set(rows)) != len(rows):
            raise ValueError(f"node {node!r} contains duplicate documents")
    data = {"format_version": FORMAT_VERSION,
            "node_to_docs": {node: [doc_ids[r] for r in rows] for node, rows in node_to_rows.items()},
            "provenance": {"source": path.name, **(provenance or {})}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


# ---- Teacher --------------------------------------------------------------------------
def parent_name(ds, cat: str) -> str:
    return (ds["whole_collection"] if cat in ("", "Root")
            else f'the "{cat}" collection')


def classify_prompt(ds, text, children, parent, glosses) -> str:
    lines = "\n".join(f"- {c}: {glosses.get(c, '')}".rstrip(": ") for c in children)
    header = (f"You are filing a document that currently sits in "
              f"{parent_name(ds, parent)} of {ds['collection_kind']}.\n")
    rules = (f"Subcategories:\n{lines}\n"
             "Rules:\n"
             "- Judge by the document's primary subject, not by topics it merely "
             "mentions.\n"
             f"- Answer NONE only if the document does not belong in "
             f"{parent_name(ds, parent)} at all, or if its primary subject fits none "
             "of the subcategories. When one subcategory plausibly covers the "
             "primary subject, choose it rather than NONE.\n"
             "Answer with the subcategory name only, or NONE.")
    return f"{header}{rules}\n\nDocument:\n{text}"


def snap_answer(raw, children):
    """Snap a free-form reply to a valid child name, or None (= NONE)."""
    s = (raw or "").strip().strip(".").strip()
    if not s:
        raise ValueError("empty teacher reply")
    for c in children:
        if s.lower() == c.lower():
            return c
    if s.lower().startswith("none"):
        return None
    for c in children:
        if c.lower() in s.lower():
            return c
    close = difflib.get_close_matches(s, list(children), n=1, cutoff=0.75)
    return close[0] if close else None


def _most_common(answers):
    counts = Counter(answers)
    top = max(counts.values())
    for a in answers:                      # ties go to the first occurrence
        if counts[a] == top:
            return a


class Teacher:
    def __init__(self, ds, llm, glosses, votes=1, chunk_chars=40000):
        self.ds = ds
        self.llm = llm
        self.glosses = glosses
        self.votes = votes
        self.chunk_chars = chunk_chars

    def classify(self, doc_key, text, children, parent):
        """Returns (child name or None, sure). With several votes the child order is shuffled per vote;
        only a strict majority counts as sure."""
        answers = []
        for v in range(self.votes):
            order = list(children)
            if self.votes > 1:
                random.Random(f"{doc_key}::{v}").shuffle(order)
            per_chunk = []
            for chunk in split_chunks(text, self.chunk_chars):
                raw = self.llm.chat([{"role": "user", "content":
                                      classify_prompt(self.ds, chunk, order, parent, self.glosses)}],
                                    temperature=0.0, max_tokens=16)[0]
                per_chunk.append(snap_answer(raw, order))
            answers.append(_most_common(per_chunk))
        counts = Counter(answers)
        best, n = max(counts.items(), key=lambda kv: kv[1])
        if n <= self.votes / 2:
            return None, False
        return best, True

    def classify_many(self, items, children, parent):
        """items = [(doc_key, text), ...]; runs concurrently."""
        groups = [[it] for it in items]
        out = self.llm.map_groups(groups, lambda it: self.classify(it[0], it[1], children, parent))
        return [g[0] for g in out]


# ---- Build ----------------------------------------------------------------------------
def build(cfg, teacher, doc_ids, texts, emb, name_emb, catalog, log=print):
    rng = random.Random(cfg["seed"])
    index = defaultdict(list)
    stats = Counter()

    def dfs(node, rows, depth):
        if depth == 0:
            index[node].extend(rows)
        children = catalog.children[node]
        if not children:
            return
        N = len(rows)
        boost = cfg["root_boost"] if depth == 0 else 1.0
        log(f"== node '{node}' depth {depth}, {N} docs")
        verdicts = {}                        # position -> (label, sure)

        def verify(positions):
            new = [p for p in positions if p not in verdicts]
            if not new:
                return
            out = teacher.classify_many([(doc_ids[rows[p]], texts[rows[p]]) for p in new],
                                        children, node)
            for p, v in zip(new, out):
                verdicts[p] = v
            stats["teacher_docs"] += len(new)

        def seeds_of(c):
            return [p for p, (lab, sure) in verdicts.items() if sure and lab == c]

        n1 = min(N, max(5, int(N * cfg["sample_rate"] * boost)))
        verify(rng.sample(range(N), n1))
        for c in children:
            missing = cfg["child_floor"] - len(seeds_of(c))
            unverified = [p for p in range(N) if p not in verdicts]
            if missing <= 0 or not unverified:
                continue
            near = sorted(unverified, key=lambda p: -float(emb[rows[p]] @ name_emb[c]))
            picks = near[:cfg["floor_pool"]]
            log(f"  child '{c}' has {len(seeds_of(c))} seeds, verifying the {len(picks)} docs nearest to its name")
            verify(picks)
        protos, cols = [], []
        for c in children:
            s = seeds_of(c)
            protos.append(normalize(emb[[rows[p] for p in s]].mean(axis=0)) if s
                          else name_emb[c])
            cols.append(c)
        stay = [p for p, (lab, sure) in verdicts.items() if sure and lab is None]
        if len(stay) >= 3:
            protos.append(normalize(emb[[rows[p] for p in stay]].mean(axis=0)))
            cols.append(None)
        scores = emb[rows] @ np.stack(protos).T
        dest = {}
        for p in range(N):
            v = verdicts.get(p)
            if v is not None and v[1]:
                dest[p] = v[0]
                continue
            j = int(np.argmax(scores[p]))
            dest[p] = None if cols[j] is None or scores[p][j] < cfg["tau"] else cols[j]
        lists = defaultdict(list)
        for p, d in dest.items():
            if d is None:
                if depth == 0:
                    lists["Others"].append(rows[p])
                    stats["others_fallback"] += 1
                else:
                    stats["stay_parent"] += 1
            else:
                lists[d].append(rows[p])
        for c, lst in lists.items():
            index[c].extend(lst)
        sure = sum(1 for v in verdicts.values() if v[1])
        extra = (f", Others:{len(lists.get('Others', []))}"
                 if depth == 0 and "Others" not in children else
                 f", stay_parent:{sum(1 for d in dest.values() if d is None)}" if depth > 0 else "")
        log(f"  teacher saw {len(verdicts)} docs ({sure} sure) -> "
            + ", ".join(f"{c}:{len(lists.get(c, []))}" for c in children) + extra)
        for c in children:
            if lists.get(c):
                dfs(c, list(lists[c]), depth + 1)

    dfs(catalog.root_name, list(range(len(doc_ids))), 0)
    log(f"done: {dict(stats)}")
    return dict(index), dict(stats)
