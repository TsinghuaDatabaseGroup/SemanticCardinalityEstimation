"""Read-only loaders: corpus, catalog tree, dataset config, queries, labels, document embeddings;
plus the query-vector table."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


class Catalog:
    """Catalog tree. Node names are unique across the tree; children[name] / parent[name];
    names are in pre-order traversal order; raw is the original JSON."""

    def __init__(self, root: dict):
        self.raw = root
        self.root_name = root["category"]
        self.children, self.parent = {}, {}
        self._walk(root, None)

    def _walk(self, node, parent):
        name = node["category"]
        if name in self.children:
            raise ValueError(f"Duplicate catalog node name: {name!r}")
        self.parent[name] = parent
        self.children[name] = [c["category"] for c in node.get("subcategories", [])]
        for c in node.get("subcategories", []):
            self._walk(c, name)

    @property
    def names(self):
        return list(self.children)

    @property
    def root_children(self):
        return self.children[self.root_name]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj, indent=None):
    """Write to a temporary file, then rename, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_catalog(path) -> Catalog:
    return Catalog(load_json(path))


def load_corpus(path):
    """documents.jsonl: one {doc_id, text} per line; line order = embedding matrix row order."""
    doc_ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                doc_ids.append(row["doc_id"])
                texts.append(row["text"])
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Duplicate doc_id in corpus")
    return doc_ids, texts


def load_labels(path):
    """labels.jsonl (one {"doc_id", "labels": [...]} per line) -> doc_id -> set of labels.
    A document satisfies predicate X iff X is in its label set. Used only by evaluation."""
    tags = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                tags[row["doc_id"]] = {x for x in row["labels"] if x}
    return tags


DATASET_KEYS = ("name", "collection", "collection_kind", "whole_collection", "catalog_owner", "root_hint", "unit",
                "doc_label_intro", "doc_label_task", "tier_names", "root_task", "axes_task",
                "judge_role", "judge_unit", "judge_ask_system", "judge_ask", "judge_ask_multi", "judge_and_rule",
                "judge_rule", "label_noun", "label_plural")
MULTI_KEYS = ("collection", "unit", "label_kind", "relevance")   # fields for the multi-node index teacher prompt
FULLTEXT_KEYS = ("document", "task", "naming", "collection", "relevance")   # fields for the full-text label extraction prompt


def check_dataset(ds):
    """Validate the dataset config section that fills the prompt templates.
    dataset.multi is required when index.mode = multi; dataset.fulltext when query_gen.fulltext.enabled."""
    missing = [k for k in DATASET_KEYS if k not in ds]
    if missing:
        raise ValueError(f"Config dataset section is missing fields: {missing}")
    if len(ds["tier_names"]) != 3:
        raise ValueError("dataset.tier_names must contain exactly 3 names (the three label tiers)")
    for sub, keys in (("multi", MULTI_KEYS), ("fulltext", FULLTEXT_KEYS)):
        if sub in ds:
            missing = [k for k in keys if k not in ds[sub]]
            if missing:
                raise ValueError(f"Config dataset.{sub} section is missing fields: {missing}")
    return ds


def load_queries(path):
    rows = load_json(path)
    for r in rows:
        if "query" not in r or "predicates" not in r:
            raise ValueError(f"{path} is missing the query or predicates field")
    return rows


def normalize(vecs):
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-9)


def doc_order_sha256(doc_ids):
    """Digest of the corpus row order: feed doc_id + "\\n" per row into SHA-256
    (same method as ordered_doc_id_sha256 in the embedding manifest)."""
    h = hashlib.sha256()
    for d in doc_ids:
        h.update((d + "\n").encode("utf-8"))
    return h.hexdigest()


def load_embeddings(path, doc_ids, manifest=None):
    """Document embedding matrix; row i corresponds to corpus line i. If a manifest exists, check that
    its row count, dimension and doc_id order digest match the corpus."""
    emb = np.load(path)
    if emb.shape[0] != len(doc_ids):
        raise ValueError(f"Embeddings have {emb.shape[0]} rows but the corpus has {len(doc_ids)} rows")
    if manifest and Path(manifest).exists():
        m = load_json(manifest)
        if (m.get("rows"), m.get("dims")) != emb.shape:
            raise ValueError(f"Embedding shape {emb.shape} does not match manifest {(m.get('rows'), m.get('dims'))}")
        if m.get("ordered_doc_id_sha256") and m["ordered_doc_id_sha256"] != doc_order_sha256(doc_ids):
            raise ValueError("Corpus doc_id order does not match ordered_doc_id_sha256 in the embedding manifest")
    return normalize(emb)


class QueryVectors:
    """Query-vector table {query text: unit vector}. Query text is embedded directly (no instruction template)
    and L2-normalized with the same model as the documents. Missing vectors are computed in batches and the
    whole table is rewritten to path (npz). With embedder=None, a missing vector raises an error."""

    def __init__(self, path, embedder=None, batch=64):
        self.path, self.embedder, self.batch = Path(path) if path else None, embedder, batch
        self.table = {}
        if self.path and self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            self.table = dict(zip(z["queries"].tolist(), normalize(z["Q"])))

    def matrix(self, texts):
        """Vector matrix in the order of texts; missing vectors are computed and saved."""
        missing = list(dict.fromkeys(q for q in texts if q not in self.table))
        if missing:
            if self.embedder is None:
                raise RuntimeError(f"{len(missing)} queries have no vector and no embedding service is set: {missing[0]!r}")
            for i in range(0, len(missing), self.batch):
                for q, v in zip(missing[i:i + self.batch], self.embedder.embed(missing[i:i + self.batch])):
                    self.table[q] = normalize(v)
            self.save()
        return np.stack([self.table[q] for q in texts]).astype(np.float32)

    def __call__(self, q):
        return self.matrix([q])[0]

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.table)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, queries=np.array(keys), Q=np.stack([self.table[k] for k in keys]).astype(np.float32))
        os.replace(tmp, self.path)
