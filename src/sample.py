"""Stratified importance sampling and bias-corrected extrapolation.
Strata = selected index nodes (first-come dedup across nodes); the budget is split in proportion to
stratum size N_i. Within a stratum, documents are drawn with replacement using weights
w_ij = s_ij / sum_k s_ik, where s_ij is the query-document cosine floored at SIM_FLOOR; each draw has
extrapolation weight 1 / (n_i * w_ij), so C_i = (1/n_i) sum_j y_ij / w_ij and C = sum_i C_i.
A stratum smaller than its budget is judged in full (weight 1). This module only produces the sampling
plan; judging and extrapolation are done by the caller.
For AND queries, the weight mode predicate_geomean replaces s_ij by the geometric mean (float64) of the
document's cosine with each predicate vector; the number of draws and random-number consumption are
unchanged. build_sample_plan(pred_vecs=...) selects this mode.
"""
import numpy as np

MIN_SHARE = 6   # minimum judgements per stratum so small strata are not allotted 0 draws
SIM_FLOOR = 1e-3   # cosine values below this are raised to this floor so every weight is strictly positive;
                   # the number of floored documents per stratum is recorded in groups[i]["floored"]
AND_WEIGHTS = ("query", "predicate_geomean")   # within-stratum weight modes for AND queries


def strata_of(node_lists):
    """Selected nodes become strata, deduplicated first-come across nodes; empty strata are skipped.
    Returns (strata, seen)."""
    strata, seen = [], set()
    for docs in node_lists:
        s = [d for d in docs if d not in seen]   # dedup across nodes so no document is counted twice
        seen.update(s)
        if s:
            strata.append(s)
    return strata, seen


def stratum_weights(s, emb, query_vec):
    """Within-stratum sampling probabilities w (float32 cosine -> floor -> normalize) and the number of
    floored documents. emb and query_vec must be unit vectors."""
    sim = emb[s] @ query_vec
    floored = int((sim < SIM_FLOOR).sum())
    sim = np.maximum(sim, SIM_FLOOR)          # weights must be strictly positive
    return sim / sim.sum(), floored


def predicate_weights(s, emb, pred_vecs):
    """Within-stratum sampling probabilities for AND queries: cosine of each document with each predicate
    vector (float32 dot product cast to float64), floored, geometric mean, normalized.
    pred_vecs has shape (n_predicates, dim)."""
    sim = (emb[s] @ pred_vecs.T).astype(np.float64)
    floored = int((sim < SIM_FLOOR).any(axis=1).sum())
    sim = np.maximum(sim, SIM_FLOOR)
    score = np.exp(np.log(sim).mean(axis=1))
    return score / score.sum(), floored


def build_sample_plan(node_lists, emb, query_vec, budget, rng, pred_vecs=None):
    """Return (groups, cand_rows). Each group is {"rows", "w"}: rows are the drawn row indices
    (with replacement, may repeat), w the extrapolation weight of each draw; fully judged groups have
    all weights 1.0. emb and query_vec must be unit vectors (dot product = cosine). rng is a
    random.Random instance; its consumption order determines reproducibility.
    If pred_vecs is not None, within-stratum weights come from predicate_weights."""
    strata, seen = strata_of(node_lists)
    if not strata:
        return [], seen
    total = sum(len(s) for s in strata)
    groups = []
    for s in strata:
        n = max(MIN_SHARE, int(budget * len(s) / total))
        if len(s) <= n:                          # stratum smaller than its budget: judge all
            groups.append({"rows": list(s), "w": [1.0] * len(s)})
            continue
        w, floored = (predicate_weights(s, emb, pred_vecs) if pred_vecs is not None
                      else stratum_weights(s, emb, query_vec))
        draws = rng.choices(range(len(s)), weights=w.tolist(), k=n)
        groups.append({"rows": [s[i] for i in draws],
                       "w": [1.0 / (n * float(w[i])) for i in draws], "floored": floored})
    return groups, seen


def extrapolate(groups, contributions_per_group):
    """Bias-corrected extrapolation: sum_i (1/n_i) sum_j y_ij / w_ij, i.e. each draw's weight times its verdict, summed."""
    est = 0.0
    for g, contribs in zip(groups, contributions_per_group):
        if len(contribs) != len(g["rows"]):
            raise ValueError("Number of contributions does not match number of sampled rows")
        est += float(np.dot(g["w"], contribs))
    return est
