"""Multi-membership index (config index.mode = multi): a document may belong to several sibling nodes.
At every node with children:
  a random sample of min(N, max(5, N * sample_rate * boost)) documents goes to a multi-label teacher
  (it lists all applicable children at once, possibly none);
  when a child has fewer than child_floor teacher positives, the floor_pool documents nearest to its name
  embedding are verified as well;
  per child: teacher positives < min_pos -> teacher positives only; teacher negatives < min_neg -> teacher
  positives plus every unqueried document (teacher negatives are kept out); otherwise an XGBoost binary
  classifier with xgb_trees trees is trained on the teacher-labelled documents and unqueried documents join
  when their probability >= xgb_threshold. Teacher-labelled documents always follow the teacher.
  Documents in no child go to Others at the root and stay in the parent at deeper levels; the parent keeps
  all its members; recurse into each child.
The teacher reply must be a JSON array of valid child names; otherwise a repair message is sent, up to 3
attempts in total, after which an error is raised.
Artifacts: index.json (same format as the single-membership index) and
index_multi/{teacher_assignments.json, nodes.json, classifiers/node_NNN.json}.
XGBoost is configured with n_estimators / n_jobs / random_state only (library defaults otherwise).
"""
import json
import random
import re
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from judge import split_chunks


def description(c):
    """Four-field description of the collection (collection / unit / label_kind / relevance)."""
    return (f"Collection: {c['collection']}.\nDocument unit: {c['unit']}.\n"
            f"Category labels describe {c['label_kind']}.\n{c['relevance']}")


def prompt(text, children, parent, glosses, context):
    """Multi-label classification prompt for one document (or chunk)."""
    options = "\n".join(f"- {name}: {glosses.get(name, '')}" for name in children)
    return (description(context) + f"\nAssign this document under catalog node {parent}.\n"
            f"Available child categories:\n{options}\n\n"
            "Evaluate every category independently under the collection's relevance rule. "
            "Select ALL applicable categories supported by the document. A document may "
            "have several discussion subjects or provision types. "
            "Categories are not mutually exclusive; do not choose only the strongest match. "
            "An incidental word alone is not enough. Use the category names exactly as given. "
            "Return only a JSON array of selected names, or [] if none applies.\n\n"
            f"Document:\n{text}")


def repair_message(children):
    return ("The response is not a JSON array containing only the available child "
            "category names. Re-evaluate all children independently under the same "
            "relevance rule. Do not return a parent name. Allowed names: "
            + json.dumps(children) + ". Return only the array, or [] if none applies.")


def parse_memberships(raw, children):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()).strip()
    values = json.loads(text)
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError(f"Teacher did not return a list of names: {raw!r}")
    invalid = set(values) - set(children)
    if invalid:
        raise ValueError(f"Teacher returned unknown categories: {sorted(invalid)}")
    selected = set(values)
    return [name for name in children if name in selected]


class MultiTeacher:
    """Multi-label teacher: classify((doc_id, text), children, parent) -> list of applicable child names
    (in children order)."""

    def __init__(self, llm, glosses, context, chunk_chars=40000, audit_path=None):
        self.llm, self.glosses, self.context = llm, glosses, context
        self.chunk_chars = chunk_chars
        self.audit_path = Path(audit_path) if audit_path else None
        self.lock = threading.Lock()
        self.format_repairs = 0

    def classify(self, item, children, parent):
        doc_id, text = item
        selected = set()
        for chunk in split_chunks(text, self.chunk_chars):
            messages = [{"role": "user", "content": prompt(chunk, children, parent, self.glosses, self.context)}]
            for attempt in range(3):
                raw = self.llm.chat(messages, temperature=0.0, max_tokens=512)[0]
                try:
                    memberships = parse_memberships(raw, children)
                    break
                except ValueError as error:
                    with self.lock:
                        self.format_repairs += 1
                        if self.audit_path:
                            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(self.audit_path, "a", encoding="utf-8") as output:
                                output.write(json.dumps({"doc_id": doc_id, "parent": parent, "raw": raw, "error": str(error),
                                                         "attempt": attempt}, ensure_ascii=False) + "\n")
                    if attempt == 2:
                        raise
                    messages += [{"role": "assistant", "content": raw},
                                 {"role": "user", "content": repair_message(children)}]
            selected.update(memberships)
        return [name for name in children if name in selected]

    def classify_many(self, items, children, parent):
        groups = self.llm.map_groups([[item] for item in items], lambda item: self.classify(item, children, parent))
        return [group[0] for group in groups]


class StubTeacher:
    """Replay teacher: answers come from a teacher_assignments table ({parent: {doc_id: [children]}});
    no model is called; a missing answer raises."""

    def __init__(self, assignments):
        self.assignments = assignments

    def classify_many(self, items, children, parent):
        table = self.assignments.get(parent, {})
        out = []
        for doc_id, _ in items:
            if doc_id not in table:
                raise KeyError(f"teacher_assignments has no answer for document {doc_id!r} under {parent!r}")
            out.append([c for c in children if c in set(table[doc_id])])
        return out


def build_sampled(cfg, teacher, doc_ids, texts, emb, name_emb, catalog, model_dir, log=print):
    """Returns (index {node: [rows]}, stats, supervision {parent: {doc_id: [children]}}, node_records)."""
    from xgboost import XGBClassifier
    rng = random.Random(cfg["seed"])
    index = {name: [] for name in catalog.names}
    stats, node_records, supervision = Counter(), {}, {}
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    def visit(parent, rows, depth):
        index[parent] = list(rows)
        children = [c for c in catalog.children[parent] if c != "Others"]
        if not rows or not children:
            return
        log(f"== node '{parent}' depth {depth}, {len(rows)} docs, sampling for the multi-label teacher")
        verdicts = {}

        def verify(positions):
            positions = [p for p in positions if p not in verdicts]
            if not positions:
                return
            answers = teacher.classify_many([(doc_ids[rows[p]], texts[rows[p]]) for p in positions], children, parent)
            assert len(positions) == len(answers)
            verdicts.update(zip(positions, answers))
            stats["teacher_documents_at_nodes"] += len(positions)

        boost = cfg["root_boost"] if depth == 0 else 1.0
        count = min(len(rows), max(5, int(len(rows) * cfg["sample_rate"] * boost)))
        verify(rng.sample(range(len(rows)), count))
        for child in children:
            positives = sum(child in names for names in verdicts.values())
            unknown = [p for p in range(len(rows)) if p not in verdicts]
            if positives < cfg["child_floor"] and unknown:
                nearest = sorted(unknown, key=lambda p: -float(emb[rows[p]] @ name_emb[child]))
                verify(nearest[:cfg["floor_pool"]])
        known = sorted(verdicts)
        unknown = [p for p in range(len(rows)) if p not in verdicts]
        train_x = emb[[rows[p] for p in known]]
        children_rows, details = {}, {}
        supervision[parent] = {doc_ids[rows[p]]: verdicts[p] for p in known}
        for child in children:
            y = np.array([child in verdicts[p] for p in known], dtype=np.int32)
            positive = [p for p in known if child in verdicts[p]]
            npos, nneg = int(y.sum()), int(len(y) - y.sum())
            if npos < cfg["min_pos"]:
                chosen, rule = positive, "few_positive_teacher_only"
            elif nneg < cfg["min_neg"]:
                # keep the teacher's explicit negatives out; extrapolate only to unqueried documents
                chosen, rule = positive + unknown, "few_negative_unknown_positive"
            else:
                classifier = XGBClassifier(n_estimators=cfg["xgb_trees"], n_jobs=cfg["xgb_jobs"],
                                           random_state=cfg["seed"], verbosity=0)
                classifier.fit(train_x, y)
                probs = classifier.predict_proba(emb[[rows[p] for p in unknown]])[:, 1] if unknown else []
                chosen = positive + [p for p, probability in zip(unknown, probs) if probability >= cfg["xgb_threshold"]]
                classifier.save_model(model_dir / f"node_{catalog.names.index(child):03d}.json")
                rule = "xgboost"
                stats["classifier_models"] += 1
            chosen = sorted(chosen)
            assert all((p in chosen) == (child in verdicts[p]) for p in known)
            children_rows[child] = [rows[p] for p in chosen]
            details[child] = {"size": len(chosen), "rule": rule, "teacher_positive": npos, "teacher_negative": nneg}
            stats["rule_" + rule] += 1
        coverage = Counter(r for members in children_rows.values() for r in members)
        unmatched = [r for r in rows if r not in coverage]
        stats["multiple_children_at_nodes"] += sum(n > 1 for n in coverage.values())
        stats["document_parent_population"] += len(rows)
        if depth == 0:
            index["Others"] = unmatched
            stats["others_fallback"] += len(unmatched)
        else:
            stats["stay_parent"] += len(unmatched)
        node_records[parent] = {"depth": depth, "documents": len(rows), "teacher_documents": len(known),
                                "unassigned": len(unmatched), "children": details}
        log("  " + ", ".join(f"{c}:{len(rs)}" for c, rs in children_rows.items()) + f"; teacher saw {len(known)} docs")
        for child, members in children_rows.items():
            visit(child, members, depth + 1)

    visit(catalog.root_name, list(range(len(doc_ids))), 0)
    for node, members in index.items():
        assert len(members) == len(set(members))
        if catalog.parent.get(node):
            assert set(members) <= set(index[catalog.parent[node]])
    stats["total_node_memberships"] = sum(map(len, index.values()))
    log(f"done: {dict(stats)}")
    return index, dict(stats), supervision, node_records
