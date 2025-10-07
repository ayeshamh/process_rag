import logging
import warnings
import time
from falkordb import FalkorDB
import os
from typing import Optional, Union, List, Dict, Any

from graphrag.embedding import EmbeddingGenerator
from .ontology import Ontology
from .source import AbstractSource
from .chat_session import ChatSession
from .attribute import AttributeType, Attribute
from .helpers import map_dict_to_cypher_properties
from .model_config import KnowledgeGraphModelConfig
from .steps.extract_data_step import ExtractDataStep
from .fixtures.prompts import GRAPH_QA_SYSTEM, GRAPH_QA_PROMPT
# Import local multi-level prompts
from .fixtures.prompts import MULTI_LEVEL_QA_SYSTEM, MULTI_LEVEL_QA_PROMPT
# from .embedding import EmbeddingGenerator  # Embeddings disabled

from langfuse import observe, get_client
from datetime import datetime
langfuse = get_client()  # Initialize Langfuse client for tracing
# set session_id in format DDMMYY-RAG-Generation
session_id =  f"default-session-{datetime.now().strftime('%Y%m%d')}"
langfuse_user_id = "default-user"  # Default user ID for Langfuse


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class KnowledgeGraph:
    """Knowledge Graph model data as a network of entities and relations
    To create one it is best to provide a ontology which will define the graph's ontology
    In addition to a set of sources from which entities and relations will be extracted.
    """

    def __init__(
        self,
        name: str,
        model_config: KnowledgeGraphModelConfig,
        ontology: Optional[Ontology] = None,
        host: Optional[str] = "127.0.0.1",
        port: Optional[int] = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        qa_system_instruction: Optional[str] = None,
        qa_prompt: Optional[str] = None,
        enable_embeddings: Optional[bool] = False,
        embedding_model: Optional[str] = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
        chunk_size: Optional[int] = 400,
        chunk_overlap: Optional[int] = 100,
        vector_index_enabled: Optional[bool] = True,
        context_window_config: Optional[Dict[str, Any]] = None,
        language: Optional[str] = "de",
        config_file_path: Optional[str] = None,
    ):
        """
        Create a knowledge graph.

        Args:
            name (str): Knowledge graph name
            model_config (KnowledgeGraphModelConfig): Model configuration for the knowledge graph
            ontology (Optional[Ontology], optional): Ontology for the knowledge graph. Defaults to None.
            host (Optional[str], optional): Host for the DB connection. Defaults to "127.0.0.1".
            port (Optional[int], optional): Port for the DB connection. Defaults to 6379.
            username (Optional[str], optional): Username for DB connection. Defaults to None.
            password (Optional[str], optional): Password for DB connection. Defaults to None.
            qa_system_instruction (Optional[str], optional): System instruction for Q&A. Defaults to None.
            qa_prompt (Optional[str], optional): Prompt for Q&A. Defaults to None.
            enable_embeddings (Optional[bool], optional): Whether to enable embedding generation. Defaults to False.
            embedding_model (Optional[str], optional): Model name for embeddings. Defaults to "T-Systems-onsite/cross-en-de-roberta-sentence-transformer".
            chunk_size (Optional[int], optional): Size of chunks for embedding. Defaults to 400.
            chunk_overlap (Optional[int], optional): Overlap between chunks. Defaults to 100.
            vector_index_enabled (Optional[bool], optional): Whether to create and use vector indices. Defaults to True.
            context_window_config (Optional[Dict[str, Any]], optional): Configuration for context window. Defaults to None.
        """
        # Initialize the class attributes
        self._name = name
        self._model_config = model_config
        self._ontology = ontology
        self.sources = {}
        self.processed_sources = set()  # Track processed sources by their path/name
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.enable_embeddings = enable_embeddings
        self.embedding_model = embedding_model  
        self.chunk_size = chunk_size                                                                   
        self.chunk_overlap = chunk_overlap  
        self.vector_index_enabled = vector_index_enabled and enable_embeddings
        self.context_window_config = context_window_config
        self.language = language
        self.config_file_path = config_file_path
        self.logger = logger # Assign module logger to instance
        
        # Initialize embedding generator if embeddings are enabled
        self.embedding_generator = None
        if self.enable_embeddings:
            self._initialize_embedding_generator()

        # Initialize other attributes
        self.failed_documents = []
        global session_id, langfuse_user_id
        session_id = os.getenv('GRAPHRAG_SESSION_ID', session_id)
        langfuse_user_id = os.getenv('LANGFUSE_USER', 'default-user')
        # Initialize the db and graph
        try:
            # Connect without credentials if none provided
            if self.username or self.password:
                self.db = FalkorDB(
                    host=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                )
            else:
                self.db = FalkorDB(
                    host=self.host,
                    port=self.port,
                )
            logger.info(f"Connected to FalkorDB at {self.host}:{self.port}")
            self.graph = self.db.select_graph(self.name)
        except Exception as e:
            error_msg = f"Failed to connect to FalkorDB at {self.host}:{self.port}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # Load processed sources from the graph
        self._load_processed_sources()

        # Create full-text indexes for better search performance (run once at setup)
        self._create_fulltext_indexes()

        # Validate system instructions (keep only QA)
        if qa_system_instruction is None:
            qa_system_instruction = GRAPH_QA_SYSTEM
        else:
            if not isinstance(qa_system_instruction, str):
                raise Exception("QA system instruction should be a string")

        if qa_prompt is None:
            qa_prompt = GRAPH_QA_PROMPT
        else:
            if "{question}" not in qa_prompt:
                raise Exception("Q&A prompt should contain {question}")
           
        # Assign the validated values
        self.qa_system_instruction = qa_system_instruction
        self.qa_prompt = qa_prompt

        # --- Schema info ---
        self.node_labels = []
        self.relationship_types = []
        self.relationship_patterns = []  # New: store relationship patterns as triplets
        self.refresh_schema()
        self.debug_schema_counts()  # Add this line


    # Attributes

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise AttributeError("Cannot modify the 'name' attribute")

    @property
    def ontology(self):
        return self._ontology

    @ontology.setter
    def ontology(self, value):
        self._ontology = value

    def list_sources(self) -> list[AbstractSource]:
        """
        List of sources associated with knowledge graph

        Returns:
            list[AbstractSource]: sources
        """

        return [s.source for s in self.sources]

    @observe(name="ProcessSourcesKG",  as_type="generation")
    def process_sources(
        self, sources: list[AbstractSource], instructions: Optional[str] = None, hide_progress: Optional[bool] = False, prompt_type: Optional[str] = "V1", language: Optional[str] = None, use_ontology: Optional[bool] = True
    ) -> None:
        """
        Add entities and relations found in sources into the knowledge-graph

        Args:
            sources (list[AbstractSource]): list of sources to extract knowledge from
            instructions (Optional[str]): Instructions for processing.
            hide_progress (Optional[bool]): hide progress bar
            prompt_type (Optional[str]): Prompt type for data extraction (V1, SG, COT). Defaults to "V1".
            language (Optional[str]): Language name for prompts ("German" or "English"). Defaults to self.language.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id) 
        if self.ontology is None:
            raise Exception("Ontology is not defined")
        # Filter out sources that have already been processed
        new_sources = []
        for source in sources:
            source_path = getattr(source, 'data_source', str(source))
            if source_path not in self.processed_sources:
                new_sources.append(source)
                logger.info(f"Adding new source: {source_path}")
            else:
                logger.info(f"Skipping already processed source: {source_path}")
        langfuse.update_current_generation(input=new_sources)
        if not new_sources:
            logger.info("No new sources to process")
            return
            
        # Create graph with new sources only
        self._create_graph_with_sources(new_sources, instructions, hide_progress, prompt_type, language, use_ontology)
        
        # Record processed sources
        for source in new_sources:
            source_path = getattr(source, 'data_source', str(source))
            self.processed_sources.add(source_path)
        
        # Embedding generation step
        if self.enable_embeddings and self.embedding_generator and new_sources:
            self._generate_embeddings_for_sources(new_sources)
        
        # Update processed sources in the graph
        if new_sources:
            self._update_processed_sources_in_graph()

        # Build node-level embeddings and index to enable semantic entity search
        if self.enable_embeddings and self.embedding_generator:
            try:
                embedded = self.embedding_generator.upsert_entity_node_embeddings(self.graph)
                if embedded > 0 and self.vector_index_enabled:
                    self.embedding_generator.create_node_vector_index(self.graph)
            except Exception as e:
                self.logger.error(f"Error building node embeddings/index: {e}")

        # Auto-build communities if none exist yet (non-blocking failure)
        try:
            count_res = self.graph.query("MATCH (c:Community) RETURN COUNT(c)")
            existing = (count_res.result_set or count_res)[0][0] if count_res else 0
            if int(existing) == 0:
                from .communities import build_communities, CommunityBuildConfig
                
                # Debug: Log what nodes are available for community building
                debug_res = self.graph.query("""
                    MATCH (n) 
                    WHERE NOT 'Community' IN labels(n) AND NOT 'EmbeddingEnabled' IN labels(n)
                    RETURN labels(n)[0] as label, COUNT(n) as count
                    ORDER BY count DESC
                """)
                self.logger.info("Available nodes for community building:")
                for row in debug_res.result_set or []:
                    self.logger.info(f"  {row[0]}: {row[1]} nodes")
                
                # Use entity-only community building (chunks not connected to entities)
                cfg = CommunityBuildConfig(
                    min_size=5,  # Lower threshold to include more communities
                    include_both_layers=False,  # Only include entity nodes
                    include_labels=["Entity"]  # Only include Entity nodes
                )
                res = build_communities(self.graph, self.embedding_generator, cfg)
                self.logger.info(
                    f"Leiden community build: created={res.communities_created}, nodes={res.nodes_in_communities}, skipped={res.skipped_small_communities}, took={res.duration_s:.2f}s"
                )
        except Exception as e:
            self.logger.warning(f"Community auto-build skipped: {e}")



    @observe(name="GenerateEmbeddingsKG", as_type="generation")
    def _generate_embeddings_for_sources(self, sources: list[AbstractSource]) -> None:
        """
        Generate and store embeddings for each new source using the EmbeddingGenerator.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id)
        lang_input = []
        lang_output = []
        if not self.embedding_generator:
            self.logger.warning("Embedding generator not initialized; skipping embedding generation.")
            return
        for source in sources:
            source_path = getattr(source, 'data_source', str(source))
            lang_input.append(source_path)
            try:
                # Load document(s) from the source
                docs = list(source.load())
                for doc in docs:
                    if not hasattr(doc, 'content') or not doc.content:
                        self.logger.warning(f"No content found in document for source: {source_path}")
                        continue
                    # Generate embeddings for the document
                    chunks = self.embedding_generator.generate_embeddings(doc.content)
                    if not chunks:
                        self.logger.warning(f"No embeddings generated for source: {source_path}")
                        continue
                    # Store embeddings in the graph
                    node_ids = self.embedding_generator.store_in_falkordb(
                        self.graph,
                        chunks,
                        source_path,
                        doc_id=getattr(doc, 'id', None)
                    )
                    lang_output.append({"source": source_path, "chunk_len": len(node_ids), "chunks": chunks})
                    self.logger.info(f"Stored {len(node_ids)} embedding chunks for source: {source_path}")
            except Exception as e:
                self.logger.error(f"Error generating embeddings for source {source_path}: {e}")
        langfuse.update_current_generation(input=lang_input, output=lang_output)
        # Ensure a vector index exists for the stored Chunk embeddings so they can be queried efficiently
        if self.vector_index_enabled:
            try:
                # This call is idempotent – it will create the index only if it does not already exist
                self.embedding_generator.create_vector_index(self.graph)
                self.logger.info("Vector index for Chunk embeddings verified/created.")
            except Exception as e:
                self.logger.error(f"Error ensuring vector index for Chunk embeddings: {e}")

    @observe(name="CreateGraphWithSourcesKG", as_type="generation")
    def _create_graph_with_sources(
        self, sources: Optional[list[AbstractSource]] = None, instructions: Optional[str] = None, hide_progress: Optional[bool] = False, prompt_type: Optional[str] = "V1", language: Optional[str] = None, use_ontology: Optional[bool] = True
    ) -> None:
        """
        Create a graph using the provided sources.
        
        Args:
            sources (Optional[list[AbstractSource]]): List of sources.
            instructions (Optional[str]): Instructions for the graph creation.
            prompt_type (Optional[str]): Prompt type for data extraction (V1, SG, COT). Defaults to "V1".
            language (Optional[str]): Language code for extraction (e.g., "en", "de"). Defaults to self.language.
        """
        from .steps.extract_data_step import ExtractDataStep
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id) 
        # Use the cypher_generation model instead of extract_data since this is what's available
        # in the KnowledgeGraphModelConfig class
        step = ExtractDataStep(
            sources=list(sources),
            ontology=self.ontology,
            model=self._model_config.qa,  # Using qa model instead
            graph=self.graph,
            hide_progress=hide_progress,
            prompt_type=prompt_type,
            language=language if language is not None else self.language,
            config_file_path=self.config_file_path,
            use_ontology=use_ontology,
        )
        langfuse.update_current_generation(input={"sources": [getattr(s, 'data_source', str(s)) for s in sources], 
                                                  "ontology": self.ontology,
                                                  })
        self.failed_documents = step.run(instructions)
                
    def delete(self) -> None:
        """
        Deletes the knowledge graph and any other related resource
        e.g. Ontology, data graphs
        """
        # List available graphs
        available_graphs = self.db.list_graphs()

        # Delete KnowledgeGraph
        if self.name in available_graphs:
            self.graph.delete()

        # Nullify all attributes
        for key in self.__dict__.keys():
            setattr(self, key, None)

    # Chat Session

    @observe(name="ChatSessionKG",  as_type="generation")
    def chat_session(self):
        """
        Create a ChatSession object for this knowledge graph with enhanced multi-level response support.
        
        Returns:
            ChatSession: A ChatSession object for this knowledge graph.
        """
        from .chat_session import ChatSession
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id) 
        # Use the enhanced multi-level QA prompts if available
         # use of Multi-Level QA system and prompt
        if not MULTI_LEVEL_QA_SYSTEM:
            raise ValueError("MULTI_LEVEL_QA_SYSTEM must be defined and non-empty.")
        qa_system = MULTI_LEVEL_QA_SYSTEM

        if not MULTI_LEVEL_QA_PROMPT:
            raise ValueError("MULTI_LEVEL_QA_PROMPT must be defined and non-empty.")
        qa_prompt = MULTI_LEVEL_QA_PROMPT

        # Always refresh schema before building the prompts to ensure {graph_schema} is current
        try:
            self.refresh_schema()
        except Exception as e:
            self.logger.error(f"Failed to refresh schema before creating chat session: {e}")
        
        # Get the current graph schema as a string
        graph_schema_string = self.get_graph_schema_string()
        
        session = ChatSession(
            model_config=self._model_config,
            ontology=self.ontology,
            graph=self.graph,
            qa_system_instruction=qa_system,
            qa_prompt=qa_prompt,
            graph_schema_string=graph_schema_string,
            embedding_generator=self.embedding_generator if self.enable_embeddings else None,
            context_window_config=self.context_window_config
        )
        return session

    # Entities

    def add_entity(self, label: str, attributes: Optional[dict] = None) -> None:
        """
        Add an entity to the knowledge graph, checking if it matches the ontology

        Args:
            label (str): entity label
            attributes (Optional[dict]): entity attributes
        """

        if attributes is None:
            attributes = {}

        self._validate_entity(label, attributes)

        # Add entity to graph
        self.graph.query(
            f"CREATE (:{label} {map_dict_to_cypher_properties(attributes)})"
        )

    # Edges

    def add_edge(
        self,
        relation: str,
        source: str,
        target: str,
        source_attr: Optional[dict] = None,
        target_attr: Optional[dict] = None,
        attributes: Optional[dict] = None,
    ) -> None:
        """
        Add an edge to the knowledge graph, checking if it matches the ontology

        Args:
            relation (str): relation label
            source (str): source entity label
            target (str): target entity label
            source_attr (Optional[dict]): Source entity attributes.
            target_attr (Optional[dict]): Target entity attributes.
            attributes (Optional[dict]): Relation attributes.
        """

        source_attr = source_attr or {}
        target_attr = target_attr or {}
        attributes = attributes or {}

        self._validate_relation(
            relation, source, target, source_attr, target_attr, attributes
        )

        # Add relation to graph
        self.graph.query(
            f"MATCH (s:{source} {map_dict_to_cypher_properties(source_attr)}) MATCH (t:{target} {map_dict_to_cypher_properties(target_attr)}) MERGE (s)-[r:{relation} {map_dict_to_cypher_properties(attributes)}]->(t)"
        )

    def _validate_entity(self, entity: str, attributes: str) -> None:
        """
        Validate if the entity exists in the ontology and check its attributes.
        
        Args:
            entity (str): Entity label.
            attributes (dict): Entity attributes.
        """
        ontology_entity = self.ontology.get_entity_with_label(entity)

        if ontology_entity is None:
            raise Exception(f"Entity {entity} not found in ontology")

        self._validate_attributes_dict(attributes, ontology_entity.attributes)

    def _validate_relation(
        self,
        relation: str,
        source: str,
        target: str,
        source_attr: Optional[dict],
        target_attr: Optional[dict],
        attributes: Optional[dict],
    ) -> None:
        """
        Validate if the relation exists in the ontology and check the attributes.

        Args:
            relation (str): Relation label.
            source (str): Source entity.
            target (str): Target entity.
            source_attr (Optional[dict]): Source attributes.
            target_attr (Optional[dict]): Target attributes.
            attributes (Optional[dict]): Relation attributes.
        """
        # Validate source
        self._validate_entity(source, source_attr)

        # Validate target
        self._validate_entity(target, target_attr)

        # Validate relation
        ontology_relation = self.ontology.get_relation_with_label(relation)

        if ontology_relation is None:
            raise Exception(f"Relation {relation} not found in ontology")

        # Validate relation
        is_source_valid = ontology_relation.source.matches(source)
        is_target_valid = ontology_relation.target.matches(target)

        if not is_source_valid:
            raise Exception(
                f"Invalid relation source '{source}'. Expected '{ontology_relation.source.label}'"
            )
        if not is_target_valid:
            raise Exception(
                f"Invalid relation target '{target}'. Expected '{ontology_relation.target.label}'"
            )

        self._validate_attributes_dict(attributes, ontology_relation.attributes)

    def _validate_attributes_dict(
        self, input_attributes: dict, attributes: list[Attribute]
    ) -> None:
        """
        Validate input attributes with the attributes defined in the ontology.

        Args:
            input_attributes (dict): Input attributes to validate.
            attributes (list[Attribute]): List of allowed attributes.
        """
        # Get all attribute names
        allowed_attributes = [attr.name for attr in attributes]

        for key in input_attributes.keys():
            if key not in allowed_attributes:
                raise Exception(f"Unknown attribute : '{key}'.")

    def _initialize_embedding_generator(self):
        """
        Initialize the embedding generator with configuration from model_config
        """
        # Use the attributes directly from the KnowledgeGraph instance (self)
        # instead of trying to get them from _model_config
        if not hasattr(self, 'embedding_model'):
            logger.warning("Embedding model name not set on KnowledgeGraph instance.")
            return
        
        embedding_model = self.embedding_model
        chunk_size = getattr(self, 'chunk_size', 500) # Use getattr for defaults
        chunk_overlap = getattr(self, 'chunk_overlap', 50)

        try:
            self.embedding_generator = EmbeddingGenerator(
                model_name=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info(f"Initialized embedding generator with model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding generator: {e}")
            # Optionally re-raise or handle the error
            raise

    def has_vector_index(self) -> dict:
        """
        Check if the knowledge graph has a vector index for semantic search
        
        Returns:
            dict: Information about the vector index if it exists
        """
        if not self.graph:
            return {"exists": False}
            
        try:
            # Check if index exists
            query = "CALL db.indexes() YIELD label, properties, options WHERE label = 'Chunk' AND 'embedding' IN properties RETURN label, properties, options"
            result = self.graph.query(query)
            
            if not result or not result.result_set or len(result.result_set) == 0:
                return {"exists": False}
                
            # Extract index information
            index_info = result.result_set[0][2]
            
            # Parse options from string if needed
            if isinstance(index_info, str):
                import json
                try:
                    index_info = json.loads(index_info)
                except:
                    pass
                    
            return {
                "exists": True,
                "dimension": index_info.get("dimension", "unknown"),
                "similarity_function": index_info.get("similarityFunction", "cosine"),
                "type": "vector"
            }
        except Exception as e:
            self.logger.error(f"Error checking vector index: {str(e)}")
            return {"exists": False, "error": str(e)}

    def _load_processed_sources(self):
        """Load the set of processed source paths from the graph."""
        try:
            query = "MERGE (m:_ProcessedSources {id: 'processed'}) ON CREATE SET m.paths = [] RETURN m.paths"
            result = self.graph.query(query)
            if result and result.result_set and result.result_set[0]:
                self.processed_sources = set(result.result_set[0][0])
                self.logger.info(f"Loaded {len(self.processed_sources)} processed sources from graph.")
            else:
                self.logger.info("No processed sources found in graph or failed to retrieve.")
                self.processed_sources = set()
        except Exception as e:
            self.logger.error(f"Error loading processed sources from graph: {e}")
            self.processed_sources = set() # Start fresh if loading fails

    def _update_processed_sources_in_graph(self):
        """Update the list of processed source paths stored in the graph."""
        try:
            paths_list = list(self.processed_sources)
            # Use MERGE to ensure the node exists, then SET the paths
            query = "MERGE (m:_ProcessedSources {id: 'processed'}) SET m.paths = $paths"
            self.graph.query(query, params={'paths': paths_list})
            self.logger.info(f"Updated processed sources list in graph with {len(paths_list)} paths.")
        except Exception as e:
            self.logger.error(f"Error updating processed sources in graph: {e}")

    def _create_fulltext_indexes(self):
        """Create full-text indexes for key labels and properties.
        Safe to call repeatedly; skips if creation unsupported or already exists.
        """
        try:
            # Discover labels dynamically from DB and create per-label fulltext indexes
            discovered_labels: list[str] = []
            try:
                res = self.graph.query("CALL db.labels()")
                rows = res.result_set if hasattr(res, "result_set") else res
                discovered_labels = [r[0] for r in (rows or []) if r and isinstance(r[0], str)]
            except Exception:
                # Fallback to a minimal default set if label discovery fails
                discovered_labels = ["Entity", "Source", "Community"]

            # Always include minimal defaults
            base_specs = [
                ("Entity", "name"), ("Entity", "description"), ("Entity", "__description__"),
                ("Source", "name"), ("Source", "description"), ("Source", "__description__"),
                ("Community", "name"), ("Community", "summary"),
            ]

            # Build per-label specs for common textual fields
            dynamic_specs = []
            for label in discovered_labels:
                # Skip internal/utility labels
                if label in {"EmbeddingEnabled", "_ProcessedSources"}:
                    continue
                dynamic_specs.extend([
                    (label, "name"), (label, "description"), (label, "__description__"),
                ])

            for label, field in base_specs + dynamic_specs:
                try:
                    self.graph.query(
                        "CALL db.idx.fulltext.createNodeIndex($label,$field)",
                        {"label": label, "field": field},
                    )
                except Exception:
                    # Ignore if unsupported or already present
                    continue
        except Exception:
            # Never fail init due to index creation
            pass
    

    # Graph Operations


    def chat(self, message, include_context=False):
        """
        Send a message to the chat session and get a structured multi-level response.
        
        Args:
            message (str): The message to send.
            include_context (bool, optional): Whether to include context in the response. Defaults to False.
            
        Returns:
            Dict[str, Any]: A structured multi-level response.
        """
        # Get or create a chat session
        if not hasattr(self, "_chat_session"):
            self._chat_session = self.chat_session()
        
        # Send the message and get the response
        response = self._chat_session.send_message(message)
        
        # Remove context from response if not requested
        if not include_context and "context" in response:
            del response["context"]
        
        return response

    

    def rebuild_communities(self, min_size: int = 5) -> dict:
        """Public method to rebuild communities on demand.

        Returns a summary dict for convenience.
        """
        try:
            from .communities import build_communities, CommunityBuildConfig
            cfg = CommunityBuildConfig(min_size=min_size)
            res = build_communities(self.graph, self.embedding_generator, cfg)
            out = {
                "communities_created": res.communities_created,
                "nodes_in_communities": res.nodes_in_communities,
                "skipped_small_communities": res.skipped_small_communities,
                "duration_s": res.duration_s,
                "errors": res.errors,
            }
            self.logger.info(f"Rebuilt communities: {out}")
            return out
        except Exception as e:
            self.logger.error(f"Rebuild communities failed: {e}")
            return {"communities_created": 0, "nodes_in_communities": 0, "skipped_small_communities": 0, "duration_s": 0.0, "errors": str(e)}

    def _get_embedding(self, text: str) -> list:
        """
        Generate an embedding for the given text.
        
        Args:
            text (str): The text to generate an embedding for.
            
        Returns:
            list: The embedding vector
        """
        if not self.embedding_generator:
            raise ValueError("Embedding generator not initialized")
            
        return self.embedding_generator.generate_query_embedding(text)



    def refresh_schema(self):
        """Refresh the schema info: node labels, relationship types, and relationship patterns."""
        self.logger.info("Refreshing schema from FalkorDB...")

        node_labels_query = "CALL db.labels()"
        rel_types_query = "CALL db.relationshipTypes()"

        try:
            node_labels_result = self.graph.query(node_labels_query)
            rel_types_result = self.graph.query(rel_types_query)
        except Exception as e:
            self.logger.error(f"Schema refresh failed: {e}")
            return

        # Filter out embedding-related labels (EmbeddingEnabled is added when embeddings are enabled)
        all_labels = [row[0] for row in node_labels_result.result_set]
        # Use the default embedding label name from NodeEmbeddingConfig
        embedding_label = "EmbeddingEnabled"  # Default from NodeEmbeddingConfig.label_for_index
        self.node_labels = [label for label in all_labels if label != embedding_label]
        self.relationship_types = [row[0] for row in rel_types_result.result_set]

        # Extract relationship patterns
        self.relationship_patterns = self._extract_relationship_patterns()

        self.logger.info(f"Found {len(self.node_labels)} node labels")
        self.logger.info(f"Found {len(self.relationship_types)} relationship types")
        self.logger.info(f"Found {len(self.relationship_patterns)} relationship patterns")

    def _extract_relationship_patterns(self):
        """
        Extract relationship patterns from the graph database.
        
        Returns:
            List[str]: List of relationship patterns in the format "(:SourceLabel)-[:RELATIONSHIP]->(:TargetLabel)"
        """
        patterns = []
        
        try:
            # Query to get all unique relationship patterns (source label, relationship type, target label)
            pattern_query = """
            MATCH (s)-[r]->(t)
            RETURN DISTINCT labels(s)[0] as source_label, type(r) as rel_type, labels(t)[0] as target_label
            ORDER BY source_label, rel_type, target_label
            """
            
            result = self.graph.query(pattern_query)
            
            for row in result.result_set:
                source_label, rel_type, target_label = row
                if source_label and rel_type and target_label:
                    pattern = f"(:{source_label})-[:{rel_type}]->(:{target_label})"
                    patterns.append(pattern)
                    
        except Exception as e:
            self.logger.error(f"Error extracting relationship patterns: {e}")
            
        return patterns

    def debug_schema_counts(self):
        """Print detailed diagnostics of node labels, relationship types, and relationship patterns from the graph."""
        self.logger.info("Running schema diagnostics...")

        try:
            node_label_result = self.graph.query("MATCH (n) RETURN DISTINCT labels(n)")
            rel_type_result = self.graph.query("MATCH ()-[r]->() RETURN DISTINCT type(r)")
            node_count_result = self.graph.query("MATCH (n) RETURN COUNT(n)")
            rel_count_result = self.graph.query("MATCH ()-[r]->() RETURN COUNT(r)")
        except Exception as e:
            self.logger.error(f"Schema diagnostics failed during query: {e}")
            return

        try:
            # Unpack labels safely (labels(n) returns a list)
            label_sets = {tuple(label_list) for (label_list,) in node_label_result.result_set}
            used_labels = {label for label_tuple in label_sets for label in label_tuple}

            used_relationships = {rel_type for (rel_type,) in rel_type_result.result_set}

            self.logger.info(f"Total distinct node labels used in data: {len(used_labels)}")
            for label in sorted(used_labels):
                self.logger.info(f"  - {label}")

            self.logger.info(f"Total distinct relationship types used in data: {len(used_relationships)}")
            for rtype in sorted(used_relationships):
                self.logger.info(f"  - {rtype}")

            self.logger.info(f"Total relationship patterns found: {len(self.relationship_patterns)}")
            for pattern in sorted(self.relationship_patterns):
                self.logger.info(f"  {pattern}")

            self.logger.info(f"Total node instances in DB: {node_count_result.result_set[0][0]}")
            self.logger.info(f"Total relationship instances in DB: {rel_count_result.result_set[0][0]}")

            # Compare against schema (self.node_labels and self.relationship_types)
            expected_labels = set(self.node_labels)
            expected_rels = set(self.relationship_types)

            missing_labels = expected_labels - used_labels
            extra_labels = used_labels - expected_labels

            missing_rels = expected_rels - used_relationships
            extra_rels = used_relationships - expected_rels

            if missing_labels:
                self.logger.warning(f"Node labels declared in schema but NOT used in graph data: {missing_labels}")
            if extra_labels:
                self.logger.warning(f"Node labels used in data but NOT declared in schema: {extra_labels}")

            if missing_rels:
                self.logger.warning(f"Relationship types declared in schema but NOT used in graph data: {missing_rels}")
            if extra_rels:
                self.logger.warning(f"Relationship types used in data but NOT declared in schema: {extra_rels}")

            self.logger.info("Schema diagnostics completed.")
        except Exception as e:
            self.logger.error(f"Schema diagnostics processing failed: {e}")

    def get_graph_schema_string(self) -> str:
        """
        Generate a formatted string representation of the graph schema for use in prompts.
        
        Returns:
            str: A formatted string containing node labels, relationship types, and relationship patterns
        """
        schema_parts = []
        
        # Attempt to enrich schema with discovered attributes to guide Cypher generation
        try:
            entity_attrs_map, rel_attrs_map = self._collect_schema_attributes(sample_size=100)
        except Exception as e:
            # Fail silently if attribute enrichment fails; use empty maps
            self.logger.warning(f"Attribute discovery for schema prompt failed: {e}")
            entity_attrs_map, rel_attrs_map = {}, {}
        
        # Add node labels section with inline attributes
        if self.node_labels:
            schema_parts.append("ENTITIES WITH ATTRIBUTES:")
            for label in sorted(self.node_labels):
                attrs = entity_attrs_map.get(label, [])
                if attrs:
                    # Only show attribute names (without types) for cleaner display
                    attr_names = ", ".join([name for name, _ in attrs])
                    schema_parts.append(f"  - {label}: {attr_names}")
                else:
                    schema_parts.append(f"  - {label}: (no attributes)")
        else:
            schema_parts.append("ENTITIES WITH ATTRIBUTES: None found")
        
        schema_parts.append("")  # Empty line for separation
        
        # Add relationship types section with inline attributes
        if self.relationship_types:
            schema_parts.append("RELATIONSHIPS WITH ATTRIBUTES:")
            for rel_type in sorted(self.relationship_types):
                attrs = rel_attrs_map.get(rel_type, [])
                if attrs:
                    # Only show attribute names (without types) for cleaner display
                    attr_names = ", ".join([name for name, _ in attrs])
                    schema_parts.append(f"  - {rel_type}: {attr_names}")
                else:
                    schema_parts.append(f"  - {rel_type}: (no attributes)")
        else:
            schema_parts.append("RELATIONSHIPS WITH ATTRIBUTES: None found")
        
        schema_parts.append("")  # Empty line for separation
        
        # Add relationship patterns section with clear examples
        if self.relationship_patterns:
            schema_parts.append("VALID PATTERNS (Source-[Relationship]->Target):")
            for pattern in sorted(self.relationship_patterns):
                schema_parts.append(f"  {pattern}")
        else:
            schema_parts.append("VALID PATTERNS (Source-[Relationship]->Target): None found")
        
        schema_parts.append("")  # Empty line for separation
        
        # Add usage examples
        schema_parts.append("USAGE EXAMPLES:")
        schema_parts.append("  Nodes: (variable:EntityLabel) e.g., (r:Resident), (f:Facility)")
        schema_parts.append("  Relationships: [:RELATIONSHIP_TYPE] e.g., [:LIVES_AT], [:PARTICIPATES_IN]")
        schema_parts.append("  Complete Pattern: (source:SourceLabel)-[:RELATIONSHIP]->(target:TargetLabel)")
        
        schema_parts.append("")  # Empty line for separation
        
        # Add summary
        schema_parts.append(f"SUMMARY: {len(self.node_labels)} Entity Types, {len(self.relationship_types)} Relationship Types, {len(self.relationship_patterns)} Valid Patterns")
        
        return "\n".join(schema_parts)

    def _collect_schema_attributes(self, sample_size: int = 100) -> tuple[dict, dict]:
        """
        Collect attribute names and canonical types for each node label and relationship type.
        Excludes the node-level embedding property 'node_embedding' from output.
        
        Returns:
            (entity_attrs_map, rel_attrs_map): Dict[label->List[(name, type)]], Dict[type->List[(name, type)]]
        """
        # Lazy import to avoid circular issues
        from .attribute import AttributeType

        exclude_props = {"node_embedding","confidence"}
        entity_attrs_map: Dict[str, List[tuple[str, str]]] = {}
        rel_attrs_map: Dict[str, List[tuple[str, str]]] = {}

        # Helper to canonize typeof strings to AttributeType.value where possible
        def to_canonical_type(type_str: str) -> str | None:
            try:
                return AttributeType.from_string(type_str).value
            except Exception:
                return None

        # Nodes: iterate known labels
        for label in self.node_labels:
            try:
                q = (
                    f"MATCH (a:{label}) "
                    f"CALL {{ WITH a RETURN [k IN keys(a) | [k, typeof(a[k])]] AS types }} "
                    f"WITH types LIMIT {sample_size} UNWIND types AS kt "
                    f"RETURN kt, count(1) ORDER BY kt[0]"
                )
                res = self.graph.query(q)
                rows = res.result_set if hasattr(res, "result_set") else res
                seen: dict[str, str] = {}
                for row in rows or []:
                    kt = row[0] if row else None
                    if not kt or len(kt) < 2:
                        continue
                    name, typ = kt[0], kt[1]
                    if not name or name in exclude_props:
                        continue
                    canonical = to_canonical_type(str(typ))
                    if not canonical:
                        continue
                    # keep first seen mapping
                    if name not in seen:
                        seen[name] = canonical
                if seen:
                    entity_attrs_map[label] = sorted(seen.items(), key=lambda x: x[0].lower())
                else:
                    entity_attrs_map[label] = []
            except Exception:
                entity_attrs_map[label] = []

        # Relationships: iterate known types
        for rtype in self.relationship_types:
            try:
                q = (
                    f"MATCH ()-[a:{rtype}]->() "
                    f"CALL {{ WITH a RETURN [k IN keys(a) | [k, typeof(a[k])]] AS types }} "
                    f"WITH types LIMIT {sample_size} UNWIND types AS kt "
                    f"RETURN kt, count(1) ORDER BY kt[0]"
                )
                res = self.graph.query(q)
                rows = res.result_set if hasattr(res, "result_set") else res
                seen: dict[str, str] = {}
                for row in rows or []:
                    kt = row[0] if row else None
                    if not kt or len(kt) < 2:
                        continue
                    name, typ = kt[0], kt[1]
                    if not name or name in exclude_props:
                        continue
                    canonical = to_canonical_type(str(typ))
                    if not canonical:
                        continue
                    if name not in seen:
                        seen[name] = canonical
                if seen:
                    rel_attrs_map[rtype] = sorted(seen.items(), key=lambda x: x[0].lower())
                else:
                    rel_attrs_map[rtype] = []
            except Exception:
                rel_attrs_map[rtype] = []

        return entity_attrs_map, rel_attrs_map


