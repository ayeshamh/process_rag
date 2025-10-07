import time
from typing import Dict, List, Optional, Tuple, Any, Iterable

from pydantic import BaseModel, Field, PositiveInt, conlist, field_validator


class CommunityBuildConfig(BaseModel):
    """
    Configuration for community detection and materialization.

    - Applies Leiden partitioning on an undirected projection of the graph.
    - Creates (:Community)-[:HAS_MEMBER]->(node) for communities with size >= min_size.
    - Optionally embeds community names and creates a vector index for fast retrieval.
    """

    min_size: PositiveInt = Field(5, description="Minimum number of members in a community to materialize")
    drop_existing: bool = Field(True, description="Whether to delete existing Community nodes before building")
    include_labels: Optional[conlist(str, min_length=1)] = Field(
        default=None,
        description="Only include nodes with these labels. If None, include all except excluded labels.")
    exclude_labels: List[str] = Field(
        default_factory=lambda: ["Community"],
        description="Exclude nodes with these labels from community detection.")
    include_both_layers: bool = Field(
        default=True,
        description="Include both entity nodes and chunk nodes in community detection for hybrid communities.")
    include_relationship_types: Optional[conlist(str, min_length=1)] = Field(
        default=None,
        description="Only include edges of these relationship types. If None, include all.")
    max_edges: Optional[int] = Field(
        default=None,
        description="Optional cap for number of edges fetched to prevent memory blowups on huge graphs.")
    name_fields: List[str] = Field(
        default_factory=lambda: ["name", "__description__"],
        description="Node properties used to derive a representative name for a community.")
    create_vector_index: bool = Field(True, description="Create/ensure vector index on Community.name_embedding")
    similarity_function: str = Field("COSINE", description="Vector index similarity function (COSINE or EUCLIDEAN)")
    index_m: PositiveInt = Field(16, description="HNSW M param")
    index_ef_construction: PositiveInt = Field(200, description="HNSW efConstruction")
    index_ef_runtime: PositiveInt = Field(10, description="HNSW efRuntime (default query ef)")

    @field_validator("similarity_function")
    def _validate_similarity(cls, v: str) -> str:
        val = v.upper()
        if val not in {"COSINE", "EUCLIDEAN"}:
            raise ValueError("similarity_function must be 'COSINE' or 'EUCLIDEAN'")
        return val


class CommunityBuildResult(BaseModel):
    communities_created: int = 0
    nodes_in_communities: int = 0
    skipped_small_communities: int = 0
    duration_s: float = 0.0
    errors: Optional[str] = None
    stats: Dict[str, Any] = Field(default_factory=dict)


def _batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """Yield lists of size up to batch_size from iterable."""
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _fetch_nodes(graph, cfg: CommunityBuildConfig) -> Dict[int, Dict[str, Any]]:
    """Return mapping: node_id -> properties (with labels and name fields)."""
    # Build label filter
    if cfg.include_labels:
        label_filters = " OR ".join([f"'{lb}' IN labels(n)" for lb in cfg.include_labels])
        where_clause = f"WHERE {label_filters}"
    else:
        # Exclude certain labels (e.g., Community, EmbeddingEnabled)
        exclusions = " AND ".join([f"NOT '{lb}' IN labels(n)" for lb in (cfg.exclude_labels or [])])
        where_clause = f"WHERE {exclusions}" if exclusions else ""

    q = f"""
    MATCH (n)
    {where_clause}
    RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props
    """
    res = graph.query(q)
    rows = res.result_set if hasattr(res, "result_set") else res

    out: Dict[int, Dict[str, Any]] = {}
    for r in rows or []:
        nid = r[0]
        labels = r[1] or []
        props = r[2] or {}
        
        # If include_both_layers is True, include both entity and chunk nodes
        if cfg.include_both_layers:
            # Include all nodes except excluded ones
            out[nid] = {"labels": labels, "props": props}
        else:
            # Original behavior - only include nodes that have name fields
            has_name = any(
                props.get(field) and isinstance(props.get(field), str) and props.get(field).strip()
                for field in cfg.name_fields
            )
            if has_name:
                out[nid] = {"labels": labels, "props": props}
    
    return out


def _fetch_edges(graph, cfg: CommunityBuildConfig, node_ids: List[int]) -> List[Tuple[int, int]]:
    """Return undirected edges (a,b) where a and b in node_ids and (a)-[r]-(b) matches type filter if provided."""
    if not node_ids:
        return []
    # Filter by relationship type if provided
    type_filter = ""
    if cfg.include_relationship_types:
        types_expr = ",".join([f"'{t}'" for t in cfg.include_relationship_types])
        type_filter = f"WHERE type(r) IN [{types_expr}]"

    limit_clause = f"LIMIT {int(cfg.max_edges)}" if cfg.max_edges else ""

    q = f"""
    MATCH (a)-[r]-(b)
    {type_filter}
    WHERE id(a) IN $ids AND id(b) IN $ids
    RETURN id(a) AS s, id(b) AS t
    {limit_clause}
    """
    res = graph.query(q, {"ids": node_ids})
    rows = res.result_set if hasattr(res, "result_set") else res
    edges: List[Tuple[int, int]] = []
    for r in rows or []:
        a, b = r[0], r[1]
        if a == b:
            continue
        # store undirected by normalized order
        s, t = (a, b) if a < b else (b, a)
        edges.append((s, t))
    return edges


def _choose_representative_name(props: Dict[str, Any], name_fields: List[str]) -> Optional[str]:
    # Try standard name fields first
    for field in name_fields:
        val = props.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    
    # For chunk nodes, try alternative fields
    if 'text' in props and isinstance(props['text'], str) and props['text'].strip():
        # Use first 50 characters of chunk text as name
        text = props['text'].strip()
        return text[:50] + "..." if len(text) > 50 else text
    
    if 'source_path' in props and isinstance(props['source_path'], str):
        # Use source path for chunks
        return f"Chunk from {props['source_path']}"
    
    # For other nodes, try common alternative fields
    for field in ['label', 'type', 'id', 'path']:
        val = props.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    
    return None




def _compute_partition(node_ids: List[int], edges: List[Tuple[int, int]]) -> Dict[int, int]:
    """
    Compute a community id for each node using Leiden algorithm.
    Returns mapping node_id -> community_id (int).
    """
    # Lazy imports to avoid mandatory dependency on non-build paths
    import networkx as nx  # type: ignore
    import leidenalg  # type: ignore
    import igraph as ig  # type: ignore

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(edges)
    if G.number_of_nodes() == 0:
        return {}
    
    # Convert NetworkX graph to igraph for Leiden algorithm
    # Create mapping from node IDs to consecutive integers
    node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
    index_to_node_id = {i: node_id for node_id, i in node_id_to_index.items()}
    
    # Create igraph with consecutive node indices
    ig_edges = [(node_id_to_index[edge[0]], node_id_to_index[edge[1]]) for edge in edges]
    ig_graph = ig.Graph(edges=ig_edges, directed=False)
    
    # Run Leiden algorithm
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition, 
                                       seed=42, n_iterations=2)
    
    # Convert back to node_id -> community_id mapping
    part: Dict[int, int] = {}
    for i, community_id in enumerate(partition.membership):
        node_id = index_to_node_id[i]
        part[node_id] = community_id
    
    return part


def build_communities(graph, embedding_generator, cfg: Optional[CommunityBuildConfig] = None) -> CommunityBuildResult:
    """
    Build community nodes and relations based on Leiden partitioning.

    - Deletes existing :Community nodes if cfg.drop_existing
    - Creates new :Community with properties {uuid, name, summary, size}
    - Creates (c)-[:HAS_MEMBER]->(n) edges
    - Sets c.name_embedding using provided embedding_generator
    - Ensures vector index on (Community, name_embedding) if cfg.create_vector_index
    """
    start = time.time()
    cfg = cfg or CommunityBuildConfig()
    result = CommunityBuildResult()

    try:
        # Optionally drop old communities
        if cfg.drop_existing:
            graph.query("MATCH (c:Community) DETACH DELETE c")

        # Fetch nodes and edges
        nodes_map = _fetch_nodes(graph, cfg)
        node_ids = list(nodes_map.keys())
        edges = _fetch_edges(graph, cfg, node_ids)

        # Partition
        partition = _compute_partition(node_ids, edges)
        if not partition:
            result.duration_s = time.time() - start
            return result

        # Group members per community id
        comm_to_nodes: Dict[int, List[int]] = {}
        for nid, cid in partition.items():
            comm_to_nodes.setdefault(cid, []).append(nid)

        # Create communities
        created = 0
        skipped = 0
        members_total = 0

        # Precompute degrees for representative selection
        deg: Dict[int, int] = {}
        for a, b in edges:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1

        for cid, members in comm_to_nodes.items():
            size = len(members)
            if size < cfg.min_size:
                skipped += 1
                continue

            # Choose representative by max degree; fallback to first
            rep = max(members, key=lambda n: deg.get(n, 0)) if members else None
            rep_props = nodes_map.get(rep or members[0], {}).get("props", {})
            center_name = _choose_representative_name(rep_props, cfg.name_fields) or f"Community-{cid}"
            summary = f"{center_name} + {max(0, size-1)} related"

            # Create community node
            create_res = graph.query(
                "CREATE (c:Community {name:$name, summary:$summary, size:$size}) RETURN id(c)",
                {"name": center_name, "summary": summary, "size": size},
            )
            comm_id = (create_res.result_set or create_res)[0][0]

            # Link members in batches
            for batch in _batched(members, 1000):
                graph.query(
                    """
                    UNWIND $ids AS nid
                    MATCH (c) WHERE id(c) = $cid
                    MATCH (n) WHERE id(n) = nid
                    MERGE (c)-[:HAS_MEMBER]->(n)
                    """,
                    {"ids": batch, "cid": comm_id},
                )

            # Embed community name
            if embedding_generator is not None:
                try:
                    texts_to_encode = [center_name, summary]
                    embeddings = embedding_generator.model.encode(texts_to_encode, convert_to_numpy=True)
        
                    # Update both embeddings in a single query
                    graph.query(
                        """
                        MATCH (c) WHERE id(c)=$cid 
                        SET c.name_embedding = vecf32($name_vec), 
                            c.summary_embedding = vecf32($summary_vec)
                        """,
                        {
                            "cid": comm_id, 
                            "name_vec": embeddings[0].tolist(),
                            "summary_vec": embeddings[1].tolist()
                        },
                    )
                except Exception:
                    # Continue even if embedding fails
                    pass

            created += 1
            members_total += size

        # Ensure vector index
        if cfg.create_vector_index:
            for field in ['name_embedding', 'summary_embedding']:
                try:
                    graph.query(
                        f"""
                        CALL db.idx.vector.createNodeIndex('Community','{field}','HNSW',$sim,$m,$efc,$efr)
                        """,
                        {
                            "sim": cfg.similarity_function,
                            "m": int(cfg.index_m),
                            "efc": int(cfg.index_ef_construction),
                            "efr": int(cfg.index_ef_runtime),
                        },
                    )
                except Exception:
                    # idempotent; ignore failures if index exists
                    pass

        result.communities_created = created
        result.nodes_in_communities = members_total
        result.skipped_small_communities = skipped
        result.duration_s = time.time() - start
        result.stats = {
            "nodes_considered": len(node_ids),
            "edges_considered": len(edges),
        }
        return result
    except Exception as e:
        result.errors = str(e)
        result.duration_s = time.time() - start
        return result


# Note: self-tests removed; use project test suite or call build_communities from integration tests.


