import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)



def community_vector(graph, embedding_generator, text: str, k: int = 30) -> List[Dict[str, Any]]:
    """Vector search over Community only if index and data exist; otherwise skip safely."""
    if not embedding_generator:
        return []
    try:
        # Check index and data existence
        chk = graph.query(
            "CALL db.indexes() YIELD label, properties WHERE label = 'Community' AND 'summary_embedding' IN properties RETURN count(*) AS c"
        )
        idx_ok = ((chk.result_set or chk)[0][0] if chk else 0) > 0
        dat = graph.query("MATCH (c:Community) WHERE c.summary_embedding IS NOT NULL RETURN count(c)")
        dat_ok = ((dat.result_set or dat)[0][0] if dat else 0) > 0
        if not (idx_ok and dat_ok):
            return []
    except Exception:
        return []

    # Run vector search
    from graphrag.embedding import preprocess_text
    preprocessed_text = preprocess_text(text)
    if not preprocessed_text:
        preprocessed_text = text
    qv = embedding_generator.model.encode(preprocessed_text, convert_to_numpy=True).tolist()
    try:
        res = graph.query(
            """
            CALL db.idx.vector.queryNodes('Community','summary_embedding',$k, vecf32($q)) YIELD node, score
            RETURN id(node) AS id, node.name AS name, node.summary AS summary, labels(node) AS labels, score
            """,
            {"k": k, "q": qv},
        )
        rows = res.result_set if hasattr(res, "result_set") else res
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for r in rows or []:
        out.append({"id": r[0], "name": r[1], "summary": r[2], "labels": r[3], "score": float(r[4])})
    return out


def expand_members(graph, community_ids: List[int], per_comm: int = 20) -> List[Dict[str, Any]]:
    if not community_ids:
        return []
    res = graph.query(
        f"""
        MATCH (c:Community)-[:HAS_MEMBER]->(n)
        WHERE id(c) IN $cids
        RETURN id(c) AS cid, id(n) AS id, labels(n) AS labels, properties(n) AS props
        LIMIT {len(community_ids) * per_comm}
        """,
        {"cids": community_ids},
    )
    rows = res.result_set if hasattr(res, "result_set") else res
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        out.append({"cid": r[0], "id": r[1], "labels": r[2], "props": r[3]})
    return out


