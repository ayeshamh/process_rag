from typing import List, Dict, Any


def dedup_by_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for it in items:
        nid = it.get("id")
        if nid in seen:
            continue
        seen.add(nid)
        out.append(it)
    return out


def rrf_two(a: List[Dict[str, Any]], b: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    scores, index = {}, {}
    for lst in (a, b):
        for rank, it in enumerate(lst, 1):
            nid = it.get("id")
            if nid is None:
                continue
            index[nid] = it
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**index[n], "rrf": s} for n, s in ranked]


def mmr_diversify(items: List[Dict[str, Any]], top_k: int = 20, lambda_: float = 0.7) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    pool = items[:]
    while pool and len(selected) < top_k:
        best, best_score = None, -1e9
        for it in pool:
            rel = float(it.get("rrf", 0.0) + it.get("score", 0.0))
            nov = 1.0
            if selected:
                same = any(set(it.get("labels", [])) & set(s.get("labels", [])) for s in selected)
                nov = 0.0 if same else 1.0
            sc = lambda_ * rel + (1 - lambda_) * nov
            if sc > best_score:
                best_score, best = sc, it
        selected.append(best)
        pool.remove(best)
    return selected


