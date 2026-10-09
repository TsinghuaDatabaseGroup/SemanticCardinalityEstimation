"""Minimal client for an OpenAI-compatible API: chat and embed, with a disk cache, bounded retries
and thread-pool concurrency.
Cache key = SHA-256(json.dumps([model, messages, temperature, max_tokens, n]))[:32]
(temperature must be the float 0.0 for keys to match).
"""
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LLM:
    def __init__(self, base_url, model, cache_path=None, workers=16,
                 timeout=600, extra_body=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.workers = workers
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        self.calls = 0
        self.hits = 0
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        key, val = json.loads(line)
                        self.cache[key] = val

    def _key(self, messages, temperature, max_tokens, n):
        blob = json.dumps([self.model, messages, temperature, max_tokens, n],
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def key_of(self, messages, temperature=0.0, max_tokens=1, n=1):
        """Expose the cache key for a request."""
        return self._key(messages, temperature, max_tokens, n)

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        last = None
        for attempt in range(4):
            req = urllib.request.Request(
                self.base_url + path, data=data,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as err:
                body = err.read().decode("utf-8", errors="replace")[:300]
                if err.code == 400:          # e.g. prompt exceeds the context window; retrying is pointless
                    raise RuntimeError(f"LLM rejected the request (400): {body}") from None
                last = f"HTTP {err.code}: {body}"
            except Exception as err:  # noqa: BLE001
                last = repr(err)
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"{self.base_url}{path} still failing after 4 retries: {last}")

    def chat(self, messages, temperature=0.0, max_tokens=1, n=1):
        """Return n reply texts; a repeated identical request is served from the cache."""
        key = self._key(messages, temperature, max_tokens, n)
        with self._lock:
            if key in self.cache:
                self.hits += 1
                return list(self.cache[key])
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens,
                   "n": n, **self.extra_body}
        out = self._post("/chat/completions", payload)
        replies = [(c["message"]["content"] or "") for c in out["choices"]]
        if not any(r.strip() for r in replies):
            raise RuntimeError("LLM returned an empty reply; not cached")
        with self._lock:
            self.calls += 1
            self.cache[key] = replies
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps([key, replies], ensure_ascii=False) + "\n")
        return replies

    def map_groups(self, groups, fn):
        """Run groups concurrently, items within a group sequentially. A group holds the requests
        for one document, so the server-side prefix cache is warmed by the first item.
        Returns a result list with the same shape as groups."""
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda g: [fn(x) for x in g], groups))

    def embed(self, texts):
        out = self._post("/embeddings", {"model": self.model, "input": list(texts)})
        rows = sorted(out["data"], key=lambda d: d["index"])
        vecs = np.asarray([d["embedding"] for d in rows], dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
