"""Judges. Interface: verdict(row, query, connector) -> 0/1
(row = corpus row index, query has query/predicates, connector = single/and/or).

- Judge: an LLM decides directly. Predicates are quoted, AND/OR predicates are a numbered list, the document
  comes before the question; dataset-specific wording comes from the dataset config section. Documents longer
  than chunk_chars are split into chunks judged separately; the verdict is 1 if any chunk is 1.
  Only a reply that is exactly "0" or "1" after strip is accepted; otherwise the reply and a format
  instruction are appended and the model is asked again (max_tokens 1 on the first try, 8 afterwards),
  up to ATTEMPTS tries. If all fail, a ValueError is raised. Repairs are counted in format_repairs and
  can be written to audit_path (jsonl).
- MLPJudge: a trained single-predicate dual-MLP. Each distinct predicate is scored once with the
  single-predicate query template: [q, d, q*d, |q-d|] goes through the network, p > threshold gives 1;
  AND takes min, OR takes max. An optional and_threshold replaces the threshold for AND queries.
"""
import json
import threading
from pathlib import Path

import numpy as np

REPAIR_MESSAGE = ("Return one decision for the entire question using only the character 0 or the character 1. "
                  "Do not enumerate categories or explain.")
ATTEMPTS = 3


def system_prompt(ds):
    """Judge system prompt template; ds = the dataset config section."""
    return (f"{ds['judge_role']}\n\n"
            f"You will be shown {ds['judge_unit']}, then asked whether {ds['judge_ask_system']} "
            f"one or more specific {ds['label_plural']}. {ds['label_noun'].capitalize()} names are always "
            "written in double quotes; treat everything inside one pair of quotes "
            f"as a single indivisible {ds['label_noun']} name.\n\n"
            f"{ds['judge_rule']}\n\n"
            "Answer with only 0 or 1.")


def make_query(predicates, connector):
    """Query template: "Is its content related to A and related to B?"."""
    conn = "or" if connector == "or" else "and"
    return "Is its content " + f" {conn} ".join(f"related to {p}" for p in predicates) + "?"


def question(ds, predicates, connector: str) -> str:
    if len(predicates) == 1:
        return (f'Question: {ds["judge_ask"]} '
                f'"{predicates[0]}"?\nAnswer with only 0 or 1.')
    lines = "\n".join(f'({i + 1}) "{p}"' for i, p in enumerate(predicates))
    if connector == "or":
        req = f"{ds['judge_ask_multi']} AT LEAST ONE of them?"
    else:
        req = f"{ds['judge_ask_multi']} ALL of them? {ds['judge_and_rule']}"
    return (f"Question: Consider these {ds['label_plural']}:\n{lines}\n{req}\n"
            f"Answer with only 0 or 1.")


def split_chunks(text: str, max_chars: int) -> list:
    """Split text into chunks of at most max_chars, preferring to break at a newline or sentence end.
    The chunks together cover the whole text."""
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            lo = start + max_chars // 2
            cut = max(text.rfind("\n", lo, end), text.rfind(". ", lo, end))
            if cut > start:
                end = cut + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def parse_binary(raw):
    """Strict binary parse: only "0" or "1" after strip counts; otherwise None."""
    s = (raw or "").strip()
    return int(s) if s in ("0", "1") else None


class Judge:
    def __init__(self, ds, llm, texts, chunk_chars=40000, audit_path=None):
        self.ds = ds
        self.llm = llm
        self.texts = texts
        self.chunk_chars = chunk_chars
        self.audit_path = Path(audit_path) if audit_path else None
        self.calls = 0
        self.unparsed = 0
        self.format_repairs = 0
        self._lock = threading.Lock()

    def messages(self, text, predicates, connector):
        return [{"role": "system", "content": system_prompt(self.ds)},
                {"role": "user", "content":
                 f"Document:\n{text}\n\n{question(self.ds, predicates, connector)}"}]

    def verdict(self, row, query, connector) -> int:
        for chunk_index, chunk in enumerate(split_chunks(self.texts[row], self.chunk_chars)):
            messages = self.messages(chunk, query["predicates"], connector)
            replies = []
            v = None
            for attempt in range(ATTEMPTS):
                raw = self.llm.chat(messages, temperature=0.0, max_tokens=1 if attempt == 0 else 8)[0]
                replies.append(raw)
                v = parse_binary(raw)
                if v is not None:
                    break
                messages = messages + [{"role": "assistant", "content": raw}, {"role": "user", "content": REPAIR_MESSAGE}]
            if v is None:
                with self._lock:
                    self.unparsed += 1
                raise ValueError(f"Judge gave no single 0/1 verdict in {ATTEMPTS} attempts (row {row}, {query['query']!r}): {replies!r}")
            with self._lock:
                self.calls += 1
                if attempt:
                    self.format_repairs += 1
                    if self.audit_path:
                        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(self.audit_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"row": row, "query": query["query"], "connector": connector, "chunk": chunk_index,
                                                "replies": replies, "verdict": v}, ensure_ascii=False) + "\n")
            if v == 1:
                return 1
        return 0


# ---- Trained dual-MLP judge (training in train.py) ------------------------------------
def features(q, d):
    """dual-MLP input [q, d, q*d, |q-d|]; q and d are unit vectors, optionally batched."""
    return np.concatenate([q, d, q * d, np.abs(q - d)], axis=-1).astype(np.float32)


def make_net(in_dim, hidden, layers, dropout):
    """Input linear layer + `layers` pre-LN residual blocks + output LN -> linear -> sigmoid."""
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(hidden)
            self.fc = nn.Linear(hidden, hidden)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop(torch.relu(self.fc(self.ln(x))))

    return nn.Sequential(nn.Linear(in_dim, hidden), *[Block() for _ in range(layers)],
                         nn.LayerNorm(hidden), nn.Linear(hidden, 1), nn.Sigmoid())


class MLPJudge:
    """Loads <model_dir>/{single.pt, meta.json}; emb is the full-corpus unit-vector matrix, qvec_of returns query vectors."""

    def __init__(self, model_dir, emb, qvec_of, and_threshold=None):
        import torch
        torch.set_num_threads(1)          # single-pair forward passes are small; the caller runs them in a thread pool
        m = json.loads((Path(model_dir) / "meta.json").read_text(encoding="utf-8"))["single"]
        net = make_net(emb.shape[1] * 4, m["hidden"], m["layers"], m["dropout"])
        net.load_state_dict(torch.load(Path(model_dir) / "single.pt", map_location="cpu", weights_only=True))
        self.net, self.thr, self.emb, self.qvec_of = net.eval(), float(m["threshold"]), emb, qvec_of
        self.and_thr = None if and_threshold is None else float(and_threshold)
        self.calls = self.unparsed = self.sub_calls = 0
        self._lock = threading.Lock()

    def score(self, predicate, row):
        import torch
        x = features(self.qvec_of(make_query([predicate], "single")), self.emb[row])
        with torch.no_grad():
            return float(self.net(torch.from_numpy(x[None])).item())

    def verdict(self, row, query, connector) -> int:
        thr = self.and_thr if (connector == "and" and self.and_thr is not None) else self.thr
        vs = [int(self.score(p, row) > thr) for p in dict.fromkeys(query["predicates"])]   # each distinct predicate once, in first-seen order
        with self._lock:
            self.calls += 1
            self.sub_calls += len(vs)
        return max(vs) if connector == "or" else min(vs)
