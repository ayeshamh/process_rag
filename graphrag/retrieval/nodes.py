import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def fulltext_nodes(graph, text: str, labels: Optional[List[str]] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """Full-text node search using index first, with safe fallback to CONTAINS.
    Uses per-label full-text indexes (FalkorDB/RediSearch style): index name = label.
    """
    out: List[Dict[str, Any]] = []

    def _query_index(index_name: str, qtext: str, lim: int) -> List[Dict[str, Any]]:
        try:
            # Use entire question without breaking into parts
            ft_query = qtext or ""
            res = graph.query(
                """
                CALL db.idx.fulltext.queryNodes($index, $q) YIELD node, score
                RETURN labels(node) AS labels, id(node) AS id, properties(node) AS props, score
                LIMIT $lim
                """,
                {"index": index_name, "q": ft_query, "lim": lim},
            )
            rows = res.result_set if hasattr(res, "result_set") else res
            arr: List[Dict[str, Any]] = []
            for r in rows or []:
                try:
                    arr.append({
                        "id": r[1],
                        "labels": r[0] or [],
                        "props": r[2] or {},
                        "name": (r[2] or {}).get("name"),
                        "description": (r[2] or {}).get("description"),
                        "score": float(r[3]) if r[3] is not None else 0.0,
                    })
                except Exception:
                    continue
            
                pass
            return arr
        except Exception:
            return []

    # Try index-backed queries first
    search_labels: List[str] = []
    if labels and isinstance(labels, list) and len(labels) > 0:
        search_labels = labels
    else:
        # Dynamically discover labels from DB; skip internal ones
        search_labels = []
        try:
            res = graph.query("CALL db.labels()")
            rows = res.result_set if hasattr(res, "result_set") else res
            for r in rows or []:
                lb = r[0] if r else None
                if isinstance(lb, str) and lb not in ["EmbeddingEnabled", "_ProcessedSources"]:
                    search_labels.append(lb)
        except Exception:
            search_labels = ["Entity", "Source"]

    # Query all labels and collect ALL results (no early stopping)
    for lb in search_labels:
        results = _query_index(lb, text, limit)
        out.extend(results)
    
    if out:
        # Deduplicate by node id (keep highest score) and rank all
        dedup: Dict[int, Dict[str, Any]] = {}
        for it in out:
            try:
                nid = int(it.get("id"))
                prev = dedup.get(nid)
                if (not prev) or float(it.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    dedup[nid] = it
            except Exception:
                continue
        # Return ALL ranked results (no artificial limit here)
        return sorted(dedup.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)

    # Fallback: simple CONTAINS scan
    logger.info("FULLTEXT_NODES_FALLBACK_CONTAINS query='%s' labels=%s", text, labels)
    # Use entire question for CONTAINS search
    escaped_text = (text or "").lower().replace("'", "\\'")
    where = f"toLower(n.name) CONTAINS '{escaped_text}' OR toLower(n.description) CONTAINS '{escaped_text}' OR toLower(n.__description__) CONTAINS '{escaped_text}'" if escaped_text else "false"
    # Use discovered labels (or provided labels) to scope fallback scans
    fallback_labels = labels if (labels and isinstance(labels, list) and len(labels) > 0) else search_labels
    if fallback_labels:
        for lb in fallback_labels:
            q = f"""
            MATCH (n:`{lb}`)
            WHERE {where}
            RETURN labels(n) AS labels, id(n) AS id, properties(n) AS props
            LIMIT {limit}
            """
            res = graph.query(q)
            rows = res.result_set if hasattr(res, "result_set") else res
            for r in rows or []:
                out.append({"id": r[1], "labels": r[0], "props": r[2], "score": 1.0})
        return out
    else:
        q = f"""
        MATCH (n)
        WHERE {where}
        RETURN labels(n) AS labels, id(n) AS id, properties(n) AS props
        LIMIT {limit}
        """
        res = graph.query(q)
        rows = res.result_set if hasattr(res, "result_set") else res
        for r in rows or []:
            out.append({"id": r[1], "labels": r[0], "props": r[2], "score": 1.0})
        return out


def vector_nodes(graph, embedding_generator, text: str, top_k: int = 30) -> List[Dict[str, Any]]:
    if not embedding_generator:
        return []
    # Use existing EmbeddingGenerator.search_similar_nodes to ensure index usage
    results = embedding_generator.search_similar_nodes(graph, text, top_k=top_k) or []
    out: List[Dict[str, Any]] = []
    for r in results:
        try:
            out.append({
                "id": int(r.id),
                "labels": list(r.labels) if r.labels else [],
                "props": r.properties or {},
                "score": float(r.score) if r.score is not None else 0.0,
            })
        except Exception:
            continue
    return out


