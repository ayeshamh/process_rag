from typing import List, Dict, Any


def bfs_1hop(graph, seed_ids: List[int], cap: int = 300) -> List[Dict[str, Any]]:
    if not seed_ids:
        return []
    res = graph.query(
        """
        MATCH (n)-[r]-(m)
        WHERE id(n) IN $ids
        RETURN id(n) AS src, type(r) AS rel, properties(r) AS rel_props,
               id(m) AS dst, labels(m) AS labels, properties(m) AS props,
               n.name AS src_name, n.description AS src_description,
               m.name AS dst_name, m.description AS dst_description
        LIMIT $cap
        """,
        {"ids": seed_ids, "cap": cap},
    )
    rows = res.result_set if hasattr(res, "result_set") else res
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        out.append({
            "src": r[0],
            "rel": r[1],
            "rel_props": r[2] or {},
            "dst": r[3],
            "labels": r[4],
            "props": r[5],
            "src_name": r[6],
            "src_description": r[7],
            "dst_name": r[8],
            "dst_description": r[9]
        })
    return out


def bfs_2hop(graph, seed_ids: List[int], per_seed: int = 6, cap: int = 180) -> List[Dict[str, Any]]:
    if not seed_ids:
        return []
    res = graph.query(
        f"""
        UNWIND $ids AS sid
        MATCH (s) WHERE id(s)=sid
        MATCH (s)-[r1]-(x)-[r2]-(m)
        WITH s, m, r1, r2 LIMIT {per_seed}
        RETURN id(s) AS src,
               'HOP2' AS rel,
               {{r1: properties(r1), r2: properties(r2)}} AS rel_props,
               id(m) AS dst,
               labels(m) AS labels,
               properties(m) AS props,
               s.name AS src_name, s.description AS src_description,
               m.name AS dst_name, m.description AS dst_description
        LIMIT {cap}
        """,
        {"ids": seed_ids},
    )
    rows = res.result_set if hasattr(res, "result_set") else res
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        out.append({
            "src": r[0],
            "rel": r[1],
            "rel_props": r[2] or {},
            "dst": r[3],
            "labels": r[4],
            "props": r[5],
            "src_name": r[6],
            "src_description": r[7],
            "dst_name": r[8],
            "dst_description": r[9]
        })
    return out


