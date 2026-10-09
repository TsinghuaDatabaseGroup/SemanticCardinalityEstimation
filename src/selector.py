"""Online node selection: the LLM reads the catalog tree and names the nodes that may contain
documents satisfying the query. exact_nodes and candidate_nodes are merged after parsing (all are
sampled and judged later); order-preserving dedup, since node order drives stratified sampling.
With cached = {query: [nodes]}, the LLM is not called for cached queries.
"""
import difflib
import json
import re
from collections import Counter


def build_prompt(catalog_root: dict, query: str) -> str:
    return f"""You are given a hierarchical document classification catalog as JSON and a query.
Identify:
1) "exact_nodes": nodes under which ALL documents satisfy the query;
2) "candidate_nodes": nodes under which SOME documents might satisfy the query and need verification.
Rules:
- Use node names exactly as written in the catalog.
- If a node is selected, do not also select its ancestors.
- When unsure about a node's scope, put it in candidate_nodes.
- Return ONLY a JSON object with these two keys, each a list of node names. No explanation.

Example catalog: {{"category": "Root", "subcategories": [{{"category": "Entertainment", "subcategories": [{{"category": "Music"}}, {{"category": "Films"}}]}}, {{"category": "Sports"}}]}}
Example query: "Is its content related to Music or Football?"
Example output: {{"exact_nodes": ["Music"], "candidate_nodes": ["Sports"]}}

Catalog: {json.dumps(catalog_root, ensure_ascii=False)}
Query: "{query}"
Output:"""


def parse_ballot(text: str):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object in the reply")
    obj = json.loads(m.group(0))
    return list(obj.get("exact_nodes", [])), list(obj.get("candidate_nodes", []))


def sanitize(nodes, all_names):
    """Keep exact name matches; recover small spelling differences by fuzzy match; drop the rest."""
    out = []
    for n in nodes:
        if not isinstance(n, str):
            continue
        if n in all_names:
            out.append(n)
        else:
            close = difflib.get_close_matches(n, all_names, n=1, cutoff=0.8)
            if close:
                out.append(close[0])
    return out


def tally(ballots, all_names, root_children):
    """Merge several ballots: an exact node needs a majority of votes, a candidate node needs one vote;
    with no ballots, fall back to the root's direct children."""
    if not ballots:
        return list(root_children)
    exact_votes, cand_votes = Counter(), Counter()
    for ex, ca in ballots:                     # order-preserving dedup within a ballot
        exact_votes.update(list(dict.fromkeys(sanitize(ex, all_names))))
        cand_votes.update(list(dict.fromkeys(sanitize(ca, all_names))))
    need = 2 if len(ballots) >= 2 else 1
    exact = [n for n, v in exact_votes.items() if v >= need]
    merged = list(exact)
    for n in list(cand_votes) + list(exact_votes):
        if n not in merged:
            merged.append(n)
    return merged


class Selector:
    def __init__(self, catalog, llm=None, votes=1, temperature=0.0, cached=None, log=print):
        self.catalog = catalog
        self.llm = llm
        self.votes = votes
        self.temperature = temperature
        self.cached = dict(cached or {})
        self.log = log

    def select(self, query: str) -> list:
        if query in self.cached:
            return list(self.cached[query])
        if self.llm is None:
            raise RuntimeError(f"Query not in the cached selections and no LLM is set: {query!r}")
        prompt = build_prompt(self.catalog.raw, query)
        replies = self.llm.chat([{"role": "user", "content": prompt}],
                                temperature=self.temperature, max_tokens=300,
                                n=self.votes)
        ballots = []
        for raw in replies:
            try:
                ballots.append(parse_ballot(raw))
            except Exception:  # noqa: BLE001  one unparsable ballot does not block the others
                pass
        if not ballots:
            self.log(f"  Node selection could not be parsed; falling back to the root's direct children: {query!r}")
        nodes = tally(ballots, self.catalog.names, self.catalog.root_children)
        self.cached[query] = nodes
        return nodes
