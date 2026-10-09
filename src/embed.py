"""Document embeddings: encode every corpus document with the embedding service and save a float32 matrix
(row i = corpus line i) plus a manifest {rows, dims, ordered_doc_id_sha256} used to check alignment on load."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from data import doc_order_sha256, write_json


def embed_corpus(texts, client, max_tokens, batch=16, workers=4, log=print):
    """Encode texts in corpus order. The server truncates each input to max_tokens."""
    idx = [list(range(i, min(i + batch, len(texts)))) for i in range(0, len(texts), batch)]
    out = [None] * len(idx)
    t0, done = time.time(), [0]

    def fn(k):
        resp = client._post("/embeddings", {"model": client.model, "input": [texts[i] for i in idx[k]],
                                            "truncate_prompt_tokens": int(max_tokens)})
        rows = sorted(resp["data"], key=lambda d: d["index"])
        out[k] = np.asarray([d["embedding"] for d in rows], dtype=np.float32)
        done[0] += 1
        if done[0] % 50 == 0:
            log(f"  embedded {done[0]}/{len(idx)} batches  {time.time() - t0:.0f}s")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fn, range(len(idx))))
    return np.concatenate(out, axis=0)


def run(doc_ids, texts, client, npy_path, manifest_path, max_tokens, log=print):
    npy_path, manifest_path = Path(npy_path), Path(manifest_path)
    if npy_path.exists():
        raise SystemExit(f"{npy_path} already exists")
    emb = embed_corpus(texts, client, max_tokens, log=log)
    if emb.shape[0] != len(doc_ids):
        raise RuntimeError(f"embedding rows {emb.shape[0]} != documents {len(doc_ids)}")
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, emb)
    write_json(manifest_path, {"file": npy_path.name, "rows": int(emb.shape[0]), "dims": int(emb.shape[1]), "dtype": "float32",
                               "model": client.model, "truncate_prompt_tokens": int(max_tokens),
                               "ordered_doc_id_sha256": doc_order_sha256(doc_ids)}, indent=1)
    log(f"embeddings {emb.shape} written to {npy_path}")
    return emb
