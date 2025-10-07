import json
import os
from datetime import datetime
from falkordb import Graph
from typing import Iterator, Dict, List, Optional, Any, Tuple
from graphrag.ontology import Ontology
from graphrag.steps.qa_step import QAStep
from .model_config import KnowledgeGraphModelConfig
from .fixtures.prompts import VERIFICATION_KEYWORDS, MULTI_LEVEL_QA_PROMPT, MULTI_LEVEL_QA_SYSTEM
import re
import logging

from langfuse import observe, get_client
from datetime import datetime
langfuse = get_client()  # Initialize Langfuse client for tracing
# set session_id in format DDMMYY-RAG-Generation
# session_id =   f"default-session-{datetime.now().strftime('%Y%m%d')}"
# langfuse_user_id = "default-user"  # Default user ID for Langfuse

logger = logging.getLogger(__name__)

CYPHER_ERROR_RES = "Sorry, I could not find the answer to your question"

def serialize_falkordb_object(obj) -> Dict[str, Any]:
    """
    Serialize FalkorDB node or edge objects into readable dictionaries.
    Args:
        obj: FalkorDB node, edge, or other object
    Returns:
        Dict[str, Any]: Serialized object with labels and properties
    """
    try:
        # Handle FalkorDB Node objects
        if hasattr(obj, 'labels') and hasattr(obj, 'properties'):
            return {
                "labels": list(obj.labels) if obj.labels else [],
                "properties": dict(obj.properties) if obj.properties else {}
            }
        # Handle FalkorDB Edge objects
        elif hasattr(obj, 'relation') and hasattr(obj, 'properties'):
            return {
                "relationship_type": obj.relation if hasattr(obj, 'relation') else "Unknown",
                "properties": dict(obj.properties) if obj.properties else {}
            }
        # Handle regular dictionaries
        elif isinstance(obj, dict):
            return obj
        # Handle other objects by converting to string
        else:
            return {"content": str(obj)}
    except Exception as e:
        logger.warning(f"Error serializing FalkorDB object: {e}")
        return {"content": str(obj), "error": f"Serialization failed: {e}"}


class MultiLevelResponse:
    """
    Enhanced multi-level response structure for ChatSession
    """
    
    def __init__(self, brief_answer: str, detailed_info: Any, sources: List[str] = None, 
                 requires_verification: bool = False, verification_reason: str = "",
                 has_gaps: bool = False, alternatives: List[Dict[str, Any]] = None):
        """
        Initialize a multi-level response
        
        Args:
            brief_answer (str): Level 1 - Concise 15-50 word answer
            detailed_info (Any): Level 2 - Detailed information (can be string or structured dict)
            sources (List[str]): Level 3 - Source references
            requires_verification (bool): Whether human verification is required
            verification_reason (str): Reason for requiring verification
            has_gaps (bool): Whether there are information gaps
            alternatives (List[Dict]): Alternative suggestions for gaps
        """
        self.brief_answer = brief_answer
        self.detailed_info = detailed_info
        self.sources = sources or []
        self.requires_verification = requires_verification
        self.verification_reason = verification_reason
        self.has_gaps = has_gaps
        self.alternatives = alternatives or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "brief_answer": self.brief_answer,
            "detailed_info": self.detailed_info,
            "sources": self.sources,
            "requires_verification": self.requires_verification,
            "verification_reason": self.verification_reason if self.requires_verification else "",
            "has_gaps": self.has_gaps,
            "alternatives": self.alternatives
        }
    
    def __str__(self) -> str:
        """String representation of the response"""
        # Simply return the detailed response for clean terminal display
        if isinstance(self.detailed_info, str):
            result = self.detailed_info
        else:
            # Try to format structured information
            result = json.dumps(self.detailed_info, indent=2, ensure_ascii=False)
        
        # Only include sources if they exist and only in a minimal format
        if self.sources:
            source_text = ", ".join(self.sources)
            result += f"\n\nSources: {source_text}"
        
        # Only show verification warning when needed
        if self.requires_verification:
            result += f"\n\nRequires verification: {self.verification_reason}"
        
        # Only include alternatives if gaps and alternatives exist
        if self.has_gaps and self.alternatives:
            alt_count = len(self.alternatives)
            result += f"\n\n{alt_count} alternative(s) found."
                
        return result

class ChatSession:
    """
    Enhanced chat session with multi-level responses, verification detection, and gap resolution
    """

    def __init__(self, model_config: KnowledgeGraphModelConfig, ontology: Ontology, graph: Graph,
                qa_system_instruction: str = "",
                qa_prompt: str = "",
                graph_schema_string: str = None,
                embedding_generator: Any = None,
                context_window_config: Any = None):
        """
        Initializes a new ChatSession object.

        Args:
            model_config (KnowledgeGraphModelConfig): The model configuration to use.
            ontology (Ontology): The ontology to use.
            graph (Graph): The graph to query.
            qa_system_instruction (str): System instruction for QA model
            qa_prompt (str): Prompt template for QA
            graph_schema_string (str): The graph schema string
            embedding_generator (Any): Optional embedding generator for vector retrieval
            context_window_config (Any): Optional context window configuration
        """
        self.model_config = model_config
        self.graph = graph
        self.ontology = ontology
        logger.info("Graph connected: %s", graph)
        session_id = os.getenv('GRAPHRAG_SESSION_ID', "Unmatched-Session")
        langfuse_user_id = os.getenv('LANGFUSE_USER', 'default-user')
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id)
        # use of Multi-Level QA system and prompt
        if not MULTI_LEVEL_QA_SYSTEM:
            raise ValueError("MULTI_LEVEL_QA_SYSTEM must be defined and non-empty.")
        qa_system_instruction = MULTI_LEVEL_QA_SYSTEM

        if not MULTI_LEVEL_QA_PROMPT:
            raise ValueError("MULTI_LEVEL_QA_PROMPT must be defined and non-empty.")
        self.qa_prompt = MULTI_LEVEL_QA_PROMPT

        # Only keep QA chat session;
        self.qa_chat_session = model_config.qa.start_chat(
                qa_system_instruction
            )
        # No more redundant storage - Context Window handles everything
        
        # Metadata to store additional information about the chat session
        self.metadata = {"last_query_execution_time": None}
        
        self.graph_schema_string = graph_schema_string
        # Optional embedding generator to enable parallel vector retrieval
        self.embedding_generator = embedding_generator
        # Cached schema lists for neighbor expansion
        self._schema_labels = None
        self._schema_relationships = None
        

    # for backward compatibility of method references (if any external code calls it)
    # but we won't expose or use it internally.
    
    def _store_rag_context_in_context_window(self, question: str, context: List[Dict], 
                                           response: str, response_type: str) -> None:
        """Store RAG context in the Context Window for future follow-ups."""
        try:
            # Store in QA chat session context
            if hasattr(self.qa_chat_session, 'add_rag_message'):
                rag_context = {
                    'results_count': len(context) if context else 0,
                    'context_summary': str(context)[:200] if context else ""
                }
                self.qa_chat_session.add_rag_message(
                    "assistant", 
                    response, 
                    question, 
                    'qa', 
                    rag_context
                )
                logger.info(f"Context window: Stored RAG context. Context summary: {rag_context.get('context_summary', '')[:1000]}...")
                
        except Exception as e:
            logger.warning(f"Could not store RAG context: {e}")
    
    def _detect_verification_needs(self, message: str, response: str) -> Tuple[bool, str]:
        """
        Detect if human verification is needed based on keywords and context
        
        Args:
            message (str): The user's message
            response (str): The generated response
            
        Returns:
            Tuple[bool, str]: (requires_verification, reason)
        """
        # First check for verification keywords in the message and response
        for keyword in VERIFICATION_KEYWORDS:
            if keyword.lower() in message.lower() or keyword.lower() in response.lower():
                return True, f"Contains verification-sensitive content related to '{keyword}'"
        
        # If no keywords found, check if the response itself indicates verification
        if isinstance(response, dict):
            requires_verification = response.get("requires_verification", False)
            verification_reason = response.get("verification_reason", "")
            if requires_verification:
                return True, verification_reason
        
        return False, ""

    def _check_for_gaps(self, message: str, context: List[Dict]) -> Tuple[bool, List[Dict]]:
        """Gap detector without Cypher fallback: report gap if all buckets are empty at callsite."""
        has = bool(context and len(context) > 0)
        return (not has, [])
    
    @observe(name="ChatSession.extract_attributes_from_message", as_type="generation")
    def _extract_attributes_from_message(self, message: str) -> Dict[str, Any]:
        """
        Extract key attributes from the user's message
        
        Args:
            message (str): The user's message
            
        Returns:
            Dict[str, Any]: Extracted attributes
        """
        # Create a default fallback attribute dictionary
        fallback_attributes = {
            "query": message,
            "question": message,
            "text": message
        }
        
        # Use the LLM to extract attributes from the message
        try:
            extraction_prompt = f"""
            Extract key attributes from this user query: "{message}"
            Return a JSON object with the attributes. Include location, type, quantity, name, 
            or any other attributes that might help find alternatives.
            
            Example response format:
            {{
                "location": "Berlin",
                "type": "hotel",
                "people": 2,
                "features": ["pool", "breakfast"],
                "query": "the original query",
                "question": "the original query",
                "text": "the original query"
            }}
            
            IMPORTANT: Always include "query", "question", and "text" fields with the original query.
            
            Extract attributes from: "{message}"
            """
            
            extraction_response = self.model_config.qa.generate(extraction_prompt)
            
            # Extract JSON from potential code blocks (handles ```json ... ``` format)
            json_content = extraction_response
            
            # Check for markdown code blocks
            code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            code_block_match = re.search(code_block_pattern, extraction_response)
            if code_block_match:
                json_content = code_block_match.group(1).strip()
            
            try:
                attributes = json.loads(json_content)
                # Ensure required fields are present
                attributes.setdefault("query", message)
                attributes.setdefault("question", message)
                attributes.setdefault("text", message)
                return attributes
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from extraction response: {extraction_response}")
                logger.warning(f"JSON decode error: {str(e)}")
                return fallback_attributes
        except Exception as e:
            # Log the error and return a fallback dictionary
            logger.warning(f"Error in attribute extraction: {str(e)}")
            return fallback_attributes
    
    def _generate_alternative_description(self, alternative: Dict, original_query: str) -> str:
        """
        Generate a description for an alternative suggestion
        
        Args:
            alternative (Dict): The alternative data
            original_query (str): The original user query
            
        Returns:
            str: Description of the alternative
        """
        # Use the QA model to generate a description of this alternative
        qa_step = QAStep(
            chat_session=self.qa_chat_session,
            qa_prompt="Generate a brief description of this alternative in relation to the original query: " +
                     f"{original_query}. Alternative data: {json.dumps(alternative, ensure_ascii=False)}. " +
                     "Explain why this might be a good alternative."
        )
        
        return qa_step.run(original_query, "", [alternative])

    @observe(name="ChatSession.format_multi_level_response", as_type="generation")
    def _format_multi_level_response(self, message: str, full_response: str, 
                                   context: List[Dict], requires_verification: bool, 
                                   verification_reason: str, has_gaps: bool,
                                   alternatives: List[Dict]) -> Dict[str, Any]:
        """
        Format the response into multiple levels
        
        Args:
            message (str): The user's message
            full_response (str): The complete response from the QA model
            context (List[Dict]): The context from the graph
            requires_verification (bool): If verification is required
            verification_reason (str): Reason for verification
            has_gaps (bool): If information gaps exist
            alternatives (List[Dict]): Alternative suggestions
            
        Returns:
            Dict[str, Any]: Multi-level response structure
        """
        try:
            # Handle string response with potential code blocks
            if not isinstance(full_response, dict) and isinstance(full_response, str):
                # Check for markdown code blocks
                json_content = full_response
                code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                code_block_match = re.search(code_block_pattern, full_response)
                if code_block_match:
                    json_content = code_block_match.group(1).strip()
                    
                response_json = json.loads(json_content)
            elif isinstance(full_response, dict):
                response_json = full_response
            else:
                raise ValueError("Response is neither a string nor a dictionary")
                
            # Extract fields from the structured response
            brief_answer = response_json.get("brief_answer", "")
            detailed_info = response_json.get("detailed_info", "")
            # Prefer extracting sources from graph context (relation source_document)
            extracted_sources = self._extract_sources_from_context(context)
            sources = extracted_sources if extracted_sources else response_json.get("sources", [])
            
            # Use the provided verification flag or the one from the response
            requires_verification = requires_verification or response_json.get("requires_verification", False)
            verification_reason = verification_reason or response_json.get("verification_reason", "")
            
            # Use the provided has_gaps flag or the one from the response
            has_gaps = has_gaps or response_json.get("has_gaps", False)
            alternatives = alternatives or response_json.get("alternatives", [])
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Log the error
            logger.warning(f"Error parsing response format: {str(e)}.")
            
            
        
        # Normalize/clean sources: basenames without extension and deduplicated
        def _normalize_source_name(name: str) -> str:
            if not isinstance(name, str):
                return str(name)
            base = os.path.basename(name)
            root, _ = os.path.splitext(base)
            return root
        if isinstance(sources, list):
            seen = set()
            normalized = []
            for s in sources:
                clean = _normalize_source_name(s)
                if clean and clean not in seen:
                    seen.add(clean)
                    normalized.append(clean)
            sources = normalized
        else:
            sources = [_normalize_source_name(sources)] if sources else []

        # Create the multi-level response structure
        response = MultiLevelResponse(
            brief_answer=brief_answer,
            detailed_info=detailed_info,
            sources=sources,
            requires_verification=requires_verification,
            verification_reason=verification_reason,
            has_gaps=has_gaps,
            alternatives=alternatives
        )
        
        return response.to_dict()

    @observe(name="ChatSession.send_message", as_type="generation")  
    def send_message(self, message: str) -> Dict[str, Any]:
        """
    Send a message to the chat session and get a multi-level response.
    Args:
        message (str): The message to send.
    Returns:
        Dict[str, Any]: A multi-level response containing brief answer, detailed info, and sources.
    """
        import traceback  # Ensure traceback is available for error logging
        logger.debug(f"🔹 Received input: {message} (type: {type(message)})")

        # Clear conversation context to prevent contamination from previous questions
        try:
            self.qa_chat_session.clear_context()
            logger.debug("🧹 Cleared conversation context to prevent contamination")
        except Exception as e:
            logger.warning(f"Failed to clear conversation context: {e}")

        try:
            # New retrieval pipeline (Approach 2): communities + nodes + traversal, no generated Cypher
            formatted_context: List[Dict[str, Any]] = []
            try:
                from graphrag.retrieval.community_search import community_vector, expand_members
                from graphrag.retrieval.nodes import fulltext_nodes, vector_nodes
                from graphrag.retrieval.bfs import bfs_1hop, bfs_2hop
                from graphrag.retrieval.fusion import rrf_two, dedup_by_id, mmr_diversify
            except Exception as _imp_err:
                logger.error(f"Retrieval modules import failed: {_imp_err}")
                raise

            # Parallel community vector and node retrieval (we'll use communities only for expansion hints)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as ex:
                fut_com_vec = ex.submit(community_vector, self.graph, self.embedding_generator, message, 15)
                fut_nod_txt = ex.submit(fulltext_nodes, self.graph, message, None, 50)
                fut_nod_vec = ex.submit(vector_nodes, self.graph, self.embedding_generator, message, 40)
                com_vec = fut_com_vec.result()
                nod_txt, nod_vec = fut_nod_txt.result(), fut_nod_vec.result()

            # Community members (optional enrichment only)
            # Retrieve top 10 communities and expand to get up to 20 members per community
            com_fused = com_vec[:10] if com_vec else []
            members = expand_members(self.graph, [c["id"] for c in com_fused], per_comm=12)
            # Rank/dedup full-text nodes and keep many for context
            nod_txt_ranked = mmr_diversify(dedup_by_id(nod_txt or []), top_k=60)

            # Context = nodes and their properties from full-text search (as requested)
            formatted_context = []
            for it in nod_txt_ranked[:80]:
                try:
                    formatted_context.append({
                        "labels": it.get("labels", []),
                        "id": it.get("id"),
                        "properties": it.get("props", {}),
                        "name": (it.get("props", {}) or {}).get("name", it.get("name", "")),
                        "score": float(it.get("score", 0.0)),
                        "search_type": "fulltext"
                    })
                except Exception:
                    continue

            entity_context: List[Dict[str, Any]] = []
            entities_raw: List[Any] = []
            # Populate entity_context strictly from vector node retrieval results
            for v in (nod_vec or [])[:60]:
                try:
                    props = v.get("props", {}) or {}
                    entity_context.append({
                        "labels": v.get("labels", []),
                        "name": props.get("name", ""),
                        "id": v.get("id"),
                        "score": float(v.get("score", 0.0)),
                        "properties": props,
                        "search_type": "vector",
                        "description": props.get("description", props.get("__description__", "")),
                        "source_document": props.get("source_document")
                    })
                except Exception:
                    continue
            
            
            chunk_context: List[Dict[str, Any]] = []
            if getattr(self, 'embedding_generator', None):
                try:
                    # Retrieve vector results in parallel; keep token budgets small here (post-trim below)
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        fut_chunks = executor.submit(self.embedding_generator.search_similar_chunks, self.graph, message, 60)
                        # We already have nod_vec above; no need to fetch entities again
                        fut_entities = None
                        chunks = fut_chunks.result()
                        entities = None
                        entities_raw = []

                    # Normalize chunk results to a compact schema
                    for row in (chunks or []):
                        try:
                            # FalkorDB driver may return raw rows; support both row objects and dicts
                            if isinstance(row, dict):
                                text = row.get('text') or row.get(0) or ''
                                source = row.get('source') or row.get(1) or ''
                                idx = row.get('chunk_index') or row.get(2) or None
                                score = row.get('similarity') or row.get(3) or None
                            else:
                                # Assume list/tuple order [text, source, chunk_index, similarity]
                                text = row[0] if len(row) > 0 else ''
                                source = row[1] if len(row) > 1 else ''
                                idx = row[2] if len(row) > 2 else None
                                score = row[3] if len(row) > 3 else None
                            chunk_context.append({
                                "text": text,
                                "source": source,
                                "chunk_index": idx,
                                "score": float(score) if score is not None else None
                            })
                        except Exception:
                            continue


                    # Deduplicate and diversify chunks: per-source cap and text dedup
                    seen_texts = set()
                    per_source_counts: Dict[str, int] = {}
                    diversified_chunks: List[Dict[str, Any]] = []
                    for item in chunk_context:
                        text_key = (item.get("text") or "").strip()
                        src = item.get("source") or "__unknown__"
                        if text_key in seen_texts:
                            continue
                        if per_source_counts.get(src, 0) >= 10:
                            continue
                        seen_texts.add(text_key)
                        per_source_counts[src] = per_source_counts.get(src, 0) + 1
                        diversified_chunks.append(item)


                    # Final caps for token budget (keep snippet context small)
                    chunk_context = diversified_chunks[:20]
                    # Keep most relevant vector entities only
                    entity_context = entity_context[:15]
                except Exception as e:
                    logger.warning(f"Vector retrieval skipped due to error: {e}")


            # Step 1c: Optional 1–2 hop neighbor expansion around all vector nodes (BFS)
            graph_neighbors: List[Dict[str, Any]] = []
            try:
                # Use all vector nodes as seeds for neighbor expansion
                seen = set()
                seed_ids = []
                seed_scores = {}  # Map seed_id -> score for relevance ordering
                for it in (nod_vec or []):
                    nid = it.get("id")
                    if nid in seen:
                        continue
                    seen.add(nid)
                    if isinstance(nid, int) or str(nid).isdigit():
                        seed_ids.append(int(nid))
                        seed_scores[int(nid)] = float(it.get("score", 0.0))

                if seed_ids:
                    graph_neighbors = bfs_1hop(self.graph, seed_ids, cap=100)
                    if len(graph_neighbors) < 80:
                        graph_neighbors += bfs_2hop(self.graph, seed_ids, per_seed=3, cap=120)
                    
                    # Sort neighbors by seed score to maintain relevance ordering
                    for neighbor in graph_neighbors:
                        src_id = neighbor.get("src")
                        if src_id in seed_scores:
                            neighbor["seed_score"] = seed_scores[src_id]
                        else:
                            neighbor["seed_score"] = 0.0
                    
                    # Sort by seed score (highest first) to maintain relevance ordering
                    graph_neighbors.sort(key=lambda x: x.get("seed_score", 0.0), reverse=True)
            except Exception as e:
                logger.warning(f"Neighbor expansion skipped due to error: {e}")

            # Enrich: ensure entity descriptions and neighbor target descriptions when available
            try:
                for e in (entity_context or []):
                    if not (e.get("description") and str(e.get("description")).strip()):
                        p = e.get("properties") or {}
                        d = p.get("description") or p.get("__description__") or p.get("summary")
                        if isinstance(d, str) and d.strip():
                            e["description"] = d
                for n in (graph_neighbors or []):
                    # bfs_* returns target props under 'props'
                    props = n.get("props") or {}
                    rel_props = n.get("rel_props") or {}
                    if isinstance(props, dict) and props:
                        d = props.get("description") or props.get("__description__") or props.get("summary")
                        if isinstance(d, str) and d.strip():
                            n.setdefault("target_description", d)
                    # Include source document from relationship props
                    if isinstance(rel_props, dict) and rel_props:
                        rel_source_doc = rel_props.get("source_document", rel_props.get("source", ""))
                        if rel_source_doc:
                            n.setdefault("rel_source_document", rel_source_doc)
            except Exception:
                pass

     

            # Step 2: Check if human verification is needed based on the message content
            requires_verification, verification_reason = self._detect_verification_needs(message, "")

            # Step 3: Check for information gaps in the context
            has_gaps, alternatives = self._check_for_gaps(message, formatted_context)

            # Step 4: Use QA model to generate a response based on context and verification needs
            qa_step = QAStep(
                chat_session=self.qa_chat_session,
                qa_prompt=self.qa_prompt
            )

            # If verification is required, provide a special prompt to the QA model
            if requires_verification:
                custom_qa_prompt = self.qa_prompt + f"\n\nIMPORTANT: This query may require human verification related to: {verification_reason}. Mark the response accordingly."
                qa_step = QAStep(
                    chat_session=self.qa_chat_session,
                    qa_prompt=custom_qa_prompt
                )
                logger.debug("🔹 Using custom QA prompt due to verification need.")
            else:
                logger.debug("🔹 Using default QA prompt.")

            # Dump ONLY the final buckets sent to QA (embedding-like fields filtered)
            try:
                def _sanitize(obj):
                    drop = {"node_embedding", "embedding", "_embedding", "vector", "embedding_vector", "name_embedding"}
                    if isinstance(obj, dict):
                        return {k: _sanitize(v) for k, v in obj.items() if k not in drop}
                    if isinstance(obj, list):
                        return [_sanitize(v) for v in obj]
                    return obj

                clean_context = _sanitize(formatted_context)
                clean_entity_context = _sanitize(entity_context or [])
                clean_graph_neighbors = _sanitize(graph_neighbors or [])
                clean_chunk_context = _sanitize(chunk_context or [])

                final_inputs = {
                    "context": clean_context,
                    "entity_context": clean_entity_context,
                    "graph_neighbors": clean_graph_neighbors,
                    "chunk_context": clean_chunk_context,
                }
                os.makedirs("logs", exist_ok=True)
                dump_path = os.path.join(
                    "logs",
                    f"qa_inputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                )
                with open(dump_path, "w", encoding="utf-8") as f:
                    json.dump(final_inputs, f, ensure_ascii=False, indent=2)
                logger.info("Saved QA inputs to %s", dump_path)
                # Human-readable log 
                try:
                    import json as _json
                    logger.info("\n================ CONTEXT SENT TO QA PROMPT ================")
                    # CHUNK CONTEXT
                    logger.info("CHUNK CONTEXT (%d items):", len(clean_chunk_context or []))
                    for i, c in enumerate(clean_chunk_context or [], start=1):
                        src = c.get("source")
                        txt = (c.get("text") or "").strip().replace("\n", " ")
                        if len(txt) > 180:
                            txt = txt[:180] + "…"
                        logger.info("[%d] Source: %s | Score: %s | Chunk: %s", i, src, c.get("score"), txt)
                    # ENTITY CONTEXT (vector nodes)
                    logger.info("\nENTITY CONTEXT (%d items):", len(clean_entity_context or []))
                    for i, e in enumerate(clean_entity_context or [], start=1):
                        labs = "/".join(e.get("labels", []))
                        source_doc = e.get("source_document", "")
                        logger.info("[%d] %s | Score: %.3f | ID: %s | Source: %s", i, labs, float(e.get("score", 0.0)), e.get("id"), source_doc)
                    # CONTEXT (full-text nodes)
                    logger.info("\nCONTEXT (%d items):", len(clean_context or []))
                    for i, n in enumerate(clean_context or [], start=1):
                        labs = "/".join(n.get("labels", []))
                        name = (n.get("properties", {}) or {}).get("name", n.get("name", ""))
                        logger.info("[%d] %s | Name: %s | Score: %.3f | ID: %s", i, labs, name, float(n.get("score", 0.0)), n.get("id"))
                    # GRAPH NEIGHBORS
                    logger.info("\nGRAPH NEIGHBORS (%d items):", len(clean_graph_neighbors or []))
                    for i, r in enumerate(clean_graph_neighbors or [], start=1):
                        try:
                            rel_source = r.get("rel_source_document", "")
                            target_source = r.get("target_source_document", "")
                            logger.info("[%d] (%s)->[%s]->(%s) | rel_source: %s | target_source: %s | rel_props: %s", i,
                                        "/".join(r.get("source_labels", [])),
                                        r.get("rel"),
                                        "/".join(r.get("target_labels", [])),
                                        rel_source,
                                        target_source,
                                        _json.dumps(r.get("rel_props", {}), ensure_ascii=False))
                        except Exception:
                            continue
                except Exception:
                    pass
            except Exception:
                pass

            qa_result = qa_step.run(
                message, clean_context,
                chunk_context=clean_chunk_context,
                entity_context=clean_entity_context,
                graph_neighbors=clean_graph_neighbors
            )
            logger.debug(f"🔹 Got answer from QA model: {qa_result}")

            # If verification wasn't already detected, check the response
            if not requires_verification:
                # Safely extract text from qa_result
                if hasattr(qa_result, "response"):
                    qa_text = qa_result.response
                elif hasattr(qa_result, "text"):
                    qa_text = qa_result.text
                elif hasattr(qa_result, "__str__"):
                    qa_text = str(qa_result)
                else:
                    qa_text = ""

                requires_verification, verification_reason = self._detect_verification_needs(message, qa_text)

            # Format the response with multiple information levels
            # Ensure full_response is a serializable string or dict
            try:
                if hasattr(qa_result, "response"):
                    qa_text = qa_result.response
                elif hasattr(qa_result, "text"):
                    qa_text = qa_result.text
                else:
                    qa_text = str(qa_result)
            except Exception as e:
                logger.error(f"⚠️ Failed to extract response text for formatting: {e}")
                qa_text = str(qa_result)

            formatted_response = self._format_multi_level_response(
                message=message,
                full_response=qa_text,  # Now this is a string, not an object
                context=formatted_context,
                requires_verification=requires_verification,
                verification_reason=verification_reason,
                has_gaps=has_gaps,
                alternatives=alternatives
            )
            logger.debug(f"🔹 Final response ready to return: {formatted_response}")

            # Store RAG context in the Context Window for future follow-ups
            self._store_rag_context_in_context_window(message, formatted_context, qa_text, "qa")

            return formatted_response

        except Exception as e:
            logger.error(f"❌ Error in send_message: {e}")
            logger.debug(traceback.format_exc())

            # Return a reasonable fallback response
            return {
                "brief_answer": "Error processing request",
                "detailed_info": f"An error occurred while processing your request: {str(e)}",
                "sources": [],
                "requires_verification": False,
                "verification_reason": "",
                "has_gaps": True,
                "alternatives": []
            }

    

    def clean_ontology_for_prompt(self, ontology: Ontology) -> str:
        """
        Clean the ontology for use in prompts.
        
        Args:
            ontology (Ontology): The ontology to clean.
            
        Returns:
            str: A cleaned ontology string.
        """
        return str(ontology)
    
    def chat(self, message: str) -> Dict[str, Any]:
        """
        Alias for send_message for compatibility.
        
        Args:
            message (str): The message to send.
            
        Returns:
            Dict[str, Any]: The response.
        """
        return self.send_message(message)
    
    @observe(name="ChatSession.extract_sources_from_context", as_type="generation")
    def _extract_sources_from_context(self, context: List[Dict]) -> List[str]:
        """
        Extract source references from the context
        
        Args:
            context (List[Dict]): The context from knowledge graph
            
        Returns:
            List[str]: List of source references
        """
        sources: List[str] = []

        def add_source(val: Any):
            if not val:
                return
            if isinstance(val, str) and val not in sources:
                sources.append(val)

        def walk(obj: Any):
            # Recursively search for source indicators on relationships and nodes
            if isinstance(obj, dict):
                # 1) Relationship property recorded during extraction
                if "source_document" in obj and obj["source_document"]:
                    add_source(obj["source_document"])

                # 2) Serialized FalkorDB nodes may include labels/properties
                labels = obj.get("labels") if isinstance(obj.get("labels"), list) else []
                props = obj.get("properties") if isinstance(obj.get("properties"), dict) else {}

                # Chunk nodes: properties.source_path
                if labels and "Chunk" in labels and props.get("source_path"):
                    add_source(props.get("source_path"))
                # Source nodes: properties.path
                if labels and "Source" in labels and props.get("path"):
                    add_source(props.get("path"))

                # Fallback: look for common filename/document-like keys
                for key in ("source", "document", "filename", "file", "path", "source_path"):
                    if key in obj and obj[key]:
                        add_source(obj[key])

                # Dive deeper
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        for item in context:
            # Prefer exact relation property 'source_document' from serialized raw data
            if isinstance(item, dict) and "raw_data" in item:
                # raw_data contains row_data with serialized nodes/edges
                walk(item["raw_data"])  # recursive search
            else:
                walk(item)

        # Keep order but deduplicate
        seen = set()
        ordered_unique = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                ordered_unique.append(s)

        return ordered_unique

    def _get_schema_whitelists(self) -> Tuple[List[str], List[str]]:
        """Return whitelists for relationship types and node labels based on current schema.
        Falls back to permissive defaults if schema cannot be retrieved."""
        try:
            if self._schema_relationships is None or self._schema_labels is None:
                # Attempt to use KnowledgeGraph schema if available on the Graph object is not exposed; rely on prompts
                # Since ChatSession does not have direct schema lists, derive from schema string when possible.
                schema_text = self.graph_schema_string or ""
                rels: List[str] = []
                labels: List[str] = []
                for line in schema_text.splitlines():
                    line = line.strip()
                    if line.startswith('- '):
                        token = line[2:].strip()
                        # Heuristic: relationship lines appear under RELATIONSHIPS header and are uppercase-ish
                        if token.isupper():
                            rels.append(token)
                        else:
                            labels.append(token)
                    if line.startswith('(:') and ')-[:' in line:
                        # Extract labels from pattern (:A)-[:R]->(:B)
                        import re as _re
                        m = _re.findall(r"\(:([^\)]+)\)", line)
                        for lab in m:
                            labels.append(lab)
                        m2 = _re.findall(r"\[:([^\]]+)\]", line)
                        for r in m2:
                            rels.append(r)
                # Deduplicate
                self._schema_relationships = sorted({r for r in rels if r}) or []
                self._schema_labels = sorted({l for l in labels if l}) or []
            return (self._schema_relationships or [], self._schema_labels or [])
        except Exception:
            return ([], [])