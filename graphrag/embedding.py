import logging
import re
import unicodedata
import os
import json
import string
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import faiss
import nltk
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    stopwords.words('german')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)
DEFAULT_NAME_FIELD = "name"
DEFAULT_DESCRIPTION_FIELD = "description"

# Load stopwords for German preprocessing
general_stopwords = set(stopwords.words('german'))
domain_stopwords = { "paragraf", "absatz", "artikel", "paragraph", "section"}
all_stopwords = general_stopwords.union(domain_stopwords)


def preprocess_text(text: str) -> str:
    """
    Preprocess text by tokenizing, removing stopwords, and normalizing.
    Based on the preprocessing patterns from the provided examples.
    """
    if not text or not text.strip():
        return ""
    
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    
    # Filter out stopwords and non-alphabetic tokens
    filtered_tokens = [word for word in tokens if word not in all_stopwords]
    return " ".join(filtered_tokens)


class NodeEmbeddingConfig(BaseModel):
    """
    Configuration for building and indexing node-level embeddings.
    """

    label_for_index: str = Field(
        default="EmbeddingEnabled",
        description="Nodes tagged with this label will be included in the node vector index.",
    )
    embedding_property: str = Field(
        default="node_embedding", description="Property name to store node vector."
    )
    text_fields: List[str] = Field(
        default_factory=lambda: [DEFAULT_NAME_FIELD, DEFAULT_DESCRIPTION_FIELD],
        description="Ordered list of node properties to concatenate for embedding text.",
    )
    min_characters: int = Field(
        default=3, ge=0, description="Minimum length of the concatenated text to consider."
    )
    batch_size: int = Field(
        default=256, gt=0, description="Batch size for encoding and upserting embeddings."
    )
    embed_missing_only: bool = Field(
        default=True,
        description="If True, only embed nodes that do not yet have an embedding property.",
    )
    create_index: bool = Field(
        default=True, description="Whether to (ensure) create the node vector index."
    )
    # Index parameters
    similarity_function: str = Field(
        default="cosine", description="'cosine' or 'euclidean'"
    )
    ef_construction: int = Field(default=200, gt=0)
    ef_runtime: int = Field(default=10, gt=0)
    m: int = Field(default=16, gt=0)


class NodeSearchResult(BaseModel):
    labels: List[str]
    name: Optional[str] = None
    description: Optional[str] = Field(default=None, alias=DEFAULT_DESCRIPTION_FIELD)
    id: int
    score: float
    properties: Optional[Dict[str, Any]] = None


class EmbeddingGenerator:
    """
    A class for generating document embeddings using sentence transformers.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        device: Optional[str] = None,
        use_semantic_chunking: Optional[bool] = True,
        use_faiss: bool = True,
        faiss_index_path: Optional[str] = None,
        faiss_metadata_path: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name (str): Name of the sentence transformer model to use.
            chunk_size (int): Maximum number of tokens per chunk.
            chunk_overlap (int): Number of tokens to overlap between chunks.
            device (Optional[str]): Device to run the model on ('cpu', 'cuda', etc.).
                                   If None, will use CUDA if available.
            use_faiss (bool): Whether to use FAISS for indexing and search.
            faiss_index_path (Optional[str]): Path to save/load FAISS index.
            faiss_metadata_path (Optional[str]): Path to save/load FAISS metadata.
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = bool(use_semantic_chunking)
        self.use_faiss = use_faiss
        self.faiss_index_path = faiss_index_path or "faiss_index.bin"
        self.faiss_metadata_path = faiss_metadata_path or "faiss_metadata.json"
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load the model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Loaded embedding model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
            
        # Initialize FAISS index if enabled
        self.faiss_index = None
        self.faiss_metadata = None
        if self.use_faiss:
            self._load_or_create_faiss_index()

    def _load_or_create_faiss_index(self):
        """Load existing FAISS index or create a new one."""
        try:
            if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_metadata_path):
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                # Read metadata explicitly as UTF-8
                with open(self.faiss_metadata_path, "r", encoding="utf-8") as f:
                    self.faiss_metadata = json.load(f)
                logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                # Create new index
                dimension = self.model.get_sentence_embedding_dimension()
                self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
                self.faiss_index.metric_type = faiss.METRIC_INNER_PRODUCT
                self.faiss_metadata = {
                    "ids": [],
                    "texts": [],
                    "source_paths": [],
                    "chunk_indices": []
                }
                logger.info(f"Created new FAISS index with dimension {dimension}")
        except Exception as e:
            logger.error(f"Error loading/creating FAISS index: {e}")
            self.faiss_index = None
            self.faiss_metadata = None

    def _save_faiss_index(self):
        """Save FAISS index and metadata to disk."""
        if self.faiss_index and self.faiss_metadata:
            try:
                faiss.write_index(self.faiss_index, self.faiss_index_path)
                # Write metadata with UTF-8 and without ASCII escaping to preserve umlauts/ß
                with open(self.faiss_metadata_path, "w", encoding="utf-8") as f:
                    json.dump(self.faiss_metadata, f, ensure_ascii=False)
                logger.info(f"Saved FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}")

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split a document into overlapping chunks.
        Prefers semantic boundaries (paragraphs, sentences) when enabled.
        """
        def _clean_text(t: str) -> str:
            # Remove page markers, image/table artifacts, and tabular lines; normalize bullets/ellipsis
            lines: List[str] = []
            for ln in (t or "").splitlines():
                s = ln.strip()
                if not s:
                    continue
                if "----------PAGE_END----------" in s:
                    continue
                if s.startswith("[Image:") or "[TABLE]" in s:
                    continue
                s = s.replace("…", "...").replace("•", "- ").replace("▪", "- ")
                lines.append(s)
            txt = " ".join(lines)
            txt = unicodedata.normalize("NFKC", txt)
            txt = re.sub(r"\s+", " ", txt).strip()
            return txt
        def split_into_paragraphs(t: str) -> List[str]:
            parts = re.split(r"\n\s*\n+", t.strip())
            cleaned = []
            for p in parts:
                lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
                if not lines:
                    continue
                cleaned.append(" ".join(lines))
            return cleaned or ([t.strip()] if t.strip() else [])

        def split_into_sentences(p: str) -> List[str]:
            # German-aware sentence boundaries, including punctuation like ; : and quotes
            return [s for s in re.split(r"(?<=[.!?])\s+", p.strip()) if s]


        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size: int = 0

        if self.use_semantic_chunking:
            text = _clean_text(text)
            paragraphs = split_into_paragraphs(text)
            for para in paragraphs:
                sentences = split_into_sentences(para)
                for sentence in sentences:
                    tokens = len(sentence.split())
                    if current_size + tokens > self.chunk_size and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        if self.chunk_overlap > 0:
                            overlap_size = 0
                            overlap_sentences: List[str] = []
                            for s in reversed(current_chunk):
                                s_tokens = len(s.split())
                                if overlap_size + s_tokens <= self.chunk_overlap:
                                    overlap_sentences.insert(0, s)
                                    overlap_size += s_tokens
                                else:
                                    break
                            current_chunk = overlap_sentences
                            current_size = overlap_size
                        else:
                            current_chunk = []
                            current_size = 0
                    current_chunk.append(sentence)
                    current_size += tokens
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks
        else:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sentence in sentences:
                tokens = len(sentence.split())
                if current_size + tokens > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    if self.chunk_overlap > 0:
                        overlap_size = 0
                        overlap_sentences: List[str] = []
                        for s in reversed(current_chunk):
                            s_tokens = len(s.split())
                            if overlap_size + s_tokens <= self.chunk_overlap:
                                overlap_sentences.insert(0, s)
                                overlap_size += s_tokens
                            else:
                                break
                        current_chunk = overlap_sentences
                        current_size = overlap_size
                    else:
                        current_chunk = []
                        current_size = 0
                current_chunk.append(sentence)
                current_size += tokens
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks

    def generate_embeddings(self, text: str) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a document by:
        1. Splitting it into chunks
        2. Preprocessing each chunk
        3. Embedding each chunk
        4. Adding to FAISS index if enabled
        
        Args:
            text (str): Document text to embed
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing:
                - 'text': The chunk text
                - 'embedding': Embedding vector as numpy array
                - 'chunk_index': Index of the chunk in the document
        """
        if not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return []
            
        # Split text into chunks
        chunks = self._split_text_into_chunks(text)
        
        if not chunks:
            logger.warning("No chunks generated from text")
            return []
            
        # Preprocess chunks
        preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
        
        # Generate embeddings for each chunk
        try:
            embeddings = self.model.encode(preprocessed_chunks, convert_to_numpy=True, show_progress_bar=False)
            embeddings = self._normalize_embeddings(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
            
        # Create result with chunks and embeddings
        result = []
        for i, (chunk, preprocessed_chunk, embedding) in enumerate(zip(chunks, preprocessed_chunks, embeddings)):
            result.append({
                "text": chunk,
                "preprocessed_text": preprocessed_chunk,
                "embedding": embedding,
                "chunk_index": i
            })
            
        logger.info(f"Generated {len(result)} chunk embeddings")
        return result
        
    def store_in_falkordb(self, graph, chunks: List[Dict[str, Any]], source_path: str, doc_id: Union[str, None] = None) -> List[int]:
        """
        Store chunks and their embeddings in FalkorDB as :Chunk nodes
        connected to the source document.
        
        Args:
            graph: FalkorDB graph instance
            chunks: List of chunk dictionaries with text, embedding, and chunk_index
            source_path: Path of the source document
            
        Returns:
            List[int]: List of node IDs created
        """
        node_ids = []
        
        # First ensure source node exists
        source_query = """
        MERGE (s:Source {path: $source_path})
        RETURN id(s) as source_id
        """
        
        try:
            source_result = graph.query(source_query, {'source_path': source_path})
            logger.info(f"Source node for {source_path} ensured")
        except Exception as e:
            logger.error(f"Error ensuring source node: {e}")
            return []
        
        for chunk in chunks:
            # Create a Cypher query that merges a Chunk node and connects it to the source document
            # This uses the source path, *document id*, and chunk index as a unique identifier
            query = """
            MATCH (s:Source {path: $source_path})
            MERGE (c:Chunk {
                source_path: $source_path,
                doc_id: $doc_id,
                chunk_index: $chunk_index
            })
            ON CREATE SET c.text = $text, c.embedding = vecf32($embedding)
            MERGE (s)-[:HAS_CHUNK]->(c)
            RETURN id(c) as node_id
            """
            
            params = {
                'source_path': source_path,
                'doc_id': doc_id if doc_id is not None else "__unknown__",
                'text': chunk['text'],
                'embedding': chunk['embedding'].tolist(),  # Convert numpy array to list
                'chunk_index': chunk['chunk_index']
            }
            
            try:
                result = graph.query(query, params)
                if result and result.result_set and len(result.result_set) > 0:
                    node_id = result.result_set[0][0]  # Get the first column of first row
                    node_ids.append(node_id)
                else:
                    logger.warning(f"Failed to get node ID for chunk {chunk['chunk_index']}")
            except Exception as e:
                logger.error(f"Error storing chunk in FalkorDB: {e}")
                
        # Add to FAISS index if enabled
        if self.use_faiss and self.faiss_index and self.faiss_metadata:
            try:
                embeddings_list = [chunk['embedding'] for chunk in chunks]
                if embeddings_list:
                    embeddings_array = np.array(embeddings_list, dtype=np.float32)
                    faiss.normalize_L2(embeddings_array)
                    self.faiss_index.add(embeddings_array)
                    
                    # Update metadata
                    for i, chunk in enumerate(chunks):
                        self.faiss_metadata["ids"].append(f"{source_path}_{chunk['chunk_index']}")
                        self.faiss_metadata["texts"].append(chunk.get('preprocessed_text', chunk['text']))
                        self.faiss_metadata["source_paths"].append(source_path)
                        self.faiss_metadata["chunk_indices"].append(chunk['chunk_index'])
                    
                    self._save_faiss_index()
                    logger.info(f"Added {len(chunks)} embeddings to FAISS index")
            except Exception as e:
                logger.error(f"Error adding embeddings to FAISS index: {e}")
        
        logger.info(f"Stored {len(node_ids)} chunks in FalkorDB for {source_path}")
        return node_ids
    
    def create_vector_index(self, graph, vector_dimension: int = None, similarity_function: str = 'cosine', 
                           ef_construction: int = 200, ef_runtime: int = 10, m: int = 16) -> bool:
        """
        Create a vector index for Chunk nodes' embedding attribute.
        
        Args:
            graph: FalkorDB graph instance
            vector_dimension: Dimension of embedding vectors (defaults to model's dimension)
            similarity_function: Similarity function to use ('cosine' or 'euclidean')
            ef_construction: Number of candidates during construction (higher = more accurate but slower)
            ef_runtime: Number of candidates during search (higher = more accurate but slower)
            m: Maximum number of outgoing edges per node in the index
            
        Returns:
            bool: Success or failure
        """
        try:
            # Get dimension from model if not provided
            if vector_dimension is None:
                vector_dimension = self.model.get_sentence_embedding_dimension()
            
            # Check if index already exists
            indexes_query = "CALL db.indexes() YIELD label, properties WHERE label = 'Chunk' AND 'embedding' IN properties RETURN count(*)"
            result = graph.query(indexes_query)
            rows = result.result_set if hasattr(result, "result_set") else result
            if rows and len(rows) > 0 and rows[0][0] > 0:
                logger.info("Vector index for Chunk.embedding already exists")
                return True
            
            # Create vector index on Chunk nodes with enhanced options
            query = """
            CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding) 
            OPTIONS {
                dimension: $dimension, 
                similarityFunction: $similarity_function,
                efConstruction: $ef_construction,
                efRuntime: $ef_runtime,
                M: $m
            }
            """
            
            params = {
                'dimension': vector_dimension,
                'similarity_function': similarity_function,
                'ef_construction': ef_construction,
                'ef_runtime': ef_runtime, 
                'm': m
            }
            
            graph.query(query, params)
            logger.info(f"Created vector index for Chunk embeddings with dimension {vector_dimension}, similarity function: {similarity_function}")
            return True
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            return False
    
    def _build_node_text(self, node_props: Dict[str, Any], fields: List[str]) -> str:
        parts: List[str] = []
        for field in fields:
            value = node_props.get(field)
            if isinstance(value, (str, int, float)):
                text = str(value).strip()
                if text:
                    parts.append(text)
        return " ".join(parts).strip()

    def create_node_vector_index(self, graph, cfg: Optional[NodeEmbeddingConfig] = None) -> bool:
        """
        Ensure a vector index exists for node-level embeddings on the configured label/property.
        """
        try:
            cfg = cfg or NodeEmbeddingConfig()
            dimension = self.model.get_sentence_embedding_dimension()
            # Check existing index
            check_q = (
                "CALL db.indexes() YIELD label, properties "
                "WHERE label = $label AND $prop IN properties RETURN count(*) AS c"
            )
            res = graph.query(check_q, {"label": cfg.label_for_index, "prop": cfg.embedding_property})
            # FalkorDB returns a QueryResult; use result_set for row access
            if hasattr(res, "result_set"):
                rows = res.result_set
            else:
                rows = res
            if rows and rows[0][0] > 0:
                logger.info(
                    f"Vector index for {cfg.label_for_index}.{cfg.embedding_property} already exists"
                )
                return True

            q = (
                f"CREATE VECTOR INDEX FOR (n:{cfg.label_for_index}) ON (n.{cfg.embedding_property}) "
                "OPTIONS { dimension: $dimension, similarityFunction: $similarity_function, "
                "efConstruction: $ef_construction, efRuntime: $ef_runtime, M: $m }"
            )
            graph.query(
                q,
                {
                    "dimension": dimension,
                    "similarity_function": cfg.similarity_function,
                    "ef_construction": cfg.ef_construction,
                    "ef_runtime": cfg.ef_runtime,
                    "m": cfg.m,
                },
            )
            logger.info(
                f"Created node vector index for {cfg.label_for_index}.{cfg.embedding_property} (dim={dimension})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create node vector index: {e}")
            return False

    def upsert_entity_node_embeddings(self, graph, cfg: Optional[NodeEmbeddingConfig] = None) -> int:
        """
        Compute and store node embeddings for all nodes that have sufficient text,
        built from configured fields (default: name + __description__). Adds the
        configured label to included nodes so they are part of the index scope.

        Returns number of nodes embedded in this run.
        """
        cfg = cfg or NodeEmbeddingConfig()
        total_embedded = 0

        skip = 0
        while True:
            where_missing = (
                f" AND n.{cfg.embedding_property} IS NULL" if cfg.embed_missing_only else ""
            )
            # Pull a batch of nodes with necessary props; we fetch all props, then build text client-side

            props_projection = ", ".join([f"n.{p} AS {p}" for p in set(cfg.text_fields)])
            q = (
                f"MATCH (n) WHERE 1=1{where_missing} "
                f"RETURN id(n) AS id, labels(n) AS labels, {props_projection} "
                f"SKIP $skip LIMIT $limit"
            )
            rows_res = graph.query(q, {"skip": skip, "limit": cfg.batch_size})
            rows = rows_res.result_set if hasattr(rows_res, "result_set") else rows_res
            if not rows:
                break

            node_ids: List[int] = []
            texts: List[str] = []

            # Rows are returned as lists; map them to dict by column order. We know first two columns
            # are id, labels; the rest correspond to text_fields in order.
            for r in rows:
                node_id = int(r[0])
                # r[1] = labels; subsequent positions map to cfg.text_fields order
                props: Dict[str, Any] = {}
                for idx, field in enumerate(cfg.text_fields, start=2):
                    if idx < len(r):
                        props[field] = r[idx]
                text = self._build_node_text(props, cfg.text_fields)
                if len(text) >= cfg.min_characters:
                    node_ids.append(node_id)
                    texts.append(text)

            if not texts:
                skip += len(rows)
                continue

            vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            for nid, emb in zip(node_ids, vectors):
                graph.query(
                    f"""
                    MATCH (n) WHERE id(n) = $id
                    SET n.{cfg.embedding_property} = vecf32($emb), n:{cfg.label_for_index}
                    RETURN id(n)
                    """,
                    {"id": nid, "emb": emb.tolist()},
                )
                total_embedded += 1

            skip += len(rows)
            logger.info(f"Embedded and upserted {total_embedded} nodes so far (processed batch={len(rows)})")

        if cfg.create_index:
            self.create_node_vector_index(graph, cfg)

        logger.info(f"Finished upserting node embeddings: total={total_embedded}")
        return total_embedded

    def search_similar_nodes(
        self,
        graph,
        query_text: str,
        top_k: int = 5,
        cfg: Optional[NodeEmbeddingConfig] = None,
    ) -> List[NodeSearchResult]:
        cfg = cfg or NodeEmbeddingConfig()
        
        # Use original query without preprocessing
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        try:
            results_res = graph.query(
                """
                CALL db.idx.vector.queryNodes($label, $prop, $k, vecf32($q)) YIELD node, score
                RETURN labels(node) AS labels, node.name AS name, id(node) AS id, properties(node) AS props, score
                """,
                {
                    "label": cfg.label_for_index,
                    "prop": cfg.embedding_property,
                    "k": top_k,
                    "q": query_embedding.tolist(),
                },
            )
            results = results_res.result_set if hasattr(results_res, "result_set") else results_res
        except Exception as e:
            logger.error(f"Vector index node search failed: {e}")
            return []

        parsed: List[NodeSearchResult] = []
        for row in (results or []):
            try:
                # Sanitize embedding-like properties
                def _sanitize(v: Any):
                    try:
                        drop = {"node_embedding", "embedding", "_embedding", "vector", "embedding_vector"}
                        if isinstance(v, dict):
                            return {k: _sanitize(val) for k, val in v.items() if k not in drop}
                        if isinstance(v, list):
                            return [_sanitize(val) for val in v]
                        return v
                    except Exception:
                        return v
                # Columns: labels, name, id, props, score
                props = _sanitize(row[3] if len(row) > 3 else {})
                score_col = row[4] if len(row) > 4 else None
                parsed.append(
                    NodeSearchResult(
                        labels=list(row[0]) if row[0] else [],
                        name=row[1] if row[1] is not None else None,
                        id=int(row[2]),
                        properties=props if isinstance(props, dict) else {},
                        score=float(score_col) if score_col is not None else 0.0,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse node search result row {row}: {e}")
        if parsed:
            logger.info(f"Found {len(parsed)} similar entity nodes using vector index")
        else:
            logger.info("No similar entity nodes found via vector index")
        return parsed

    def _search_chunks_with_faiss(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using FAISS index with preprocessed query.
        Internal method used by search_similar_chunks().
        
        Args:
            query_text: Query text to search for
            top_k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar chunks with similarity scores
        """
        if not self.use_faiss or not self.faiss_index or not self.faiss_metadata:
            logger.warning("FAISS index not available, falling back to graph search")
            return []
            
        if not query_text.strip():
            logger.warning("Empty query text provided")
            return []
            
        try:
            # Generate query embedding from original text
            query_embedding = self.model.encode([query_text], convert_to_numpy=True)
            query_embedding = self._normalize_embeddings(query_embedding)
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
            
            results = []
            if len(indices) > 0 and len(indices[0]) > 0:
                for idx, dist in zip(indices[0], distances[0]):
                    if idx < len(self.faiss_metadata["ids"]):
                        results.append({
                            "id": self.faiss_metadata["ids"][idx],
                            "text": self.faiss_metadata["texts"][idx],
                            "source": self.faiss_metadata["source_paths"][idx],
                            "chunk_index": self.faiss_metadata["chunk_indices"][idx],
                            "similarity": float(dist)
                        })
            
            logger.info(f"Found {len(results)} similar chunks using FAISS")
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            return []

    def search_similar_chunks(self, graph, query_text: str, top_k: int = 5, 
                             similarity_function: str = None) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a query text using FAISS index first, then fallback to graph vector index.
        
        Args:
            graph: FalkorDB graph instance
            query_text: Query text to search for
            top_k: Number of results to return
            similarity_function: Override the similarity function (None = use the one from the index)
            
        Returns:
            List[Dict[str, Any]]: List of similar chunks with similarity scores
        """
        # Try FAISS search first if available
        if self.use_faiss and self.faiss_index and self.faiss_metadata:
            faiss_results = self._search_chunks_with_faiss(query_text, top_k)
            if faiss_results:
                return faiss_results
        
        # Generate embedding for the original query
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        
        # Check if vector index exists
        index_exists = False
        index_info = {}
        try:
            index_query = "CALL db.indexes() YIELD label, properties, options WHERE label = 'Chunk' AND 'embedding' IN properties RETURN options"
            index_result = graph.query(index_query)
            index_rows = index_result.result_set if hasattr(index_result, "result_set") else index_result
            if index_rows and len(index_rows) > 0:
                index_exists = True
                index_info = index_rows[0][0]
                logger.debug(f"Found vector index with options: {index_info}")
        except Exception as e:
            logger.warning(f"Error checking index: {e}")
        
        # Try using vector index first
        if index_exists:
            try:
                # Use vector indexing with db.idx.vector.queryNodes
                query = """
                CALL db.idx.vector.queryNodes(
                    'Chunk',
                    'embedding',
                    $top_k,
                    vecf32($query_embedding)
                ) YIELD node, score
                RETURN node.text AS text, 
                       node.source_path AS source, 
                       node.chunk_index AS chunk_index, 
                       score AS similarity
                """
                params = {
                    'query_embedding': query_embedding.tolist(),
                    'top_k': top_k
                }
                result = graph.query(query, params)
                rows = result.result_set if hasattr(result, "result_set") else result
                parsed: List[Dict[str, Any]] = []
                for r in rows or []:
                    try:
                        parsed.append({
                            'text': r[0],
                            'source': r[1],
                            'chunk_index': r[2],
                            'similarity': float(r[3]) if r[3] is not None else None
                        })
                    except Exception:
                        continue
                if parsed:
                    logger.info(f"Found {len(parsed)} similar chunks using vector index")
                    return parsed
                logger.warning("No results from vector index search, falling back to manual similarity")
            except Exception as e:
                logger.warning(f"Vector index search failed: {e}. Falling back to manual similarity calculation.")
        else:
            logger.warning("No vector index found, using fallback similarity calculation")
        
        # Determine which similarity function to use for fallback
        if not similarity_function:
            if isinstance(index_info, dict) and 'similarityFunction' in index_info:
                similarity_function = index_info.get('similarityFunction', 'cosine')
            else:
                similarity_function = 'cosine'
        
        # No fallback available - return empty results if vector index fails
        logger.warning("Vector index search failed and no fallback available")
        return []

        

    def generate_query_embedding(self, text: str) -> list:
        """
        Generate an embedding for a query string with preprocessing.
        
        Args:
            text (str): The query text to embed
            
        Returns:
            list: The embedding vector as a list
        """
        if not text or not text.strip():
            logger.warning("Empty query text provided for embedding generation")
            return []
            
        try:
            # Generate embedding for the original text
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()  # Convert numpy array to list for database compatibility
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return [] 