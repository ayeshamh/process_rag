import os
import time
import json
import logging
from tqdm import tqdm
from uuid import uuid4
from datetime import datetime  # NEW: for timestamped log filenames
from falkordb import Graph
from threading import Lock
from typing import Optional
from ..steps.Step import Step
from ..document import Document
from ratelimit import limits, sleep_and_retry
from ..source import AbstractSource
from ..models.model import OutputMethod
from concurrent.futures import Future, ThreadPoolExecutor
from ..helpers import parse_ndjson, map_dict_to_cypher_properties
from ..ontology import Ontology, Entity, Relation
from ..models import (
    GenerativeModel,
    GenerativeModelChatSession,
    GenerationResponse,
    FinishReason,
)
from ..fixtures.prompts import (
    FIX_JSON_PROMPT,
    COMPLETE_DATA_EXTRACTION,
)
from ..fixtures.prompt_selector import PromptSelector
from ..steps.create_ontology_step import CreateOntologyStep  # restored for spaCy hint extraction
from collections import Counter  # NEW: for unknown label statistics
from pathlib import Path  # for robust file handling

from langfuse import observe, get_client
from datetime import datetime
langfuse = get_client()  # Initialize Langfuse client for tracing
# set session_id in format DDMMYY-RAG-Generation
session_id =   f"default-session-{datetime.now().strftime('%Y%m%d')}"
langfuse_user_id = "default-user"  # Default user ID for Langfuse

RENDER_STEP_SIZE = 0.5

# Configure logger with default level of INFO instead of DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(logging.NullHandler())

# Add a filter to suppress Cypher query logging
class CypherQueryFilter(logging.Filter):
    def filter(self, record):
        # Suppress logs that contain "Query:" which are the Cypher queries 
        if "Query:" in record.getMessage():
            return False
        return True

logger.addFilter(CypherQueryFilter())

class ExtractDataStep(Step):
    """
    Extract Data Step
    """

    def __init__(
        self,
        sources: list[AbstractSource],
        ontology: Ontology,
        model: GenerativeModel,
        graph: Graph,
        config: Optional[dict] = None,
        hide_progress: Optional[bool] = False,
        prompt_type: Optional[str] = "V1",
        language: Optional[str] = "GERMAN",
        config_file_path: Optional[str] = None,
        use_ontology: Optional[bool] = True,
    ) -> None:
        """
        Initialize the ExtractDataStep.
        
        Args:
            sources (list[AbstractSource]): List of data sources to process.
            ontology (Ontology): The ontology associated with the knowledge graph.
            model (GenerativeModel): The generative model used for data extraction.
            graph (Graph): The FalkorDB graph instance.
            config (Optional[dict]): Configuration options for the step.
            hide_progress (Optional[bool]): Flag to hide progress bar. Defaults to False.
            prompt_type (Optional[str]): Prompt type for data extraction (V1, SG, COT). Defaults to "V1".
            language (Optional[str]): Language name for prompts ("German" or "English"). Defaults to "GERMAN".
            config_file_path (Optional[str]): Path to config file for tracking skipped files. Defaults to None.
        """
        global session_id, langfuse_user_id
        session_id = os.getenv('GRAPHRAG_SESSION_ID', session_id)
        langfuse_user_id = os.getenv('LANGFUSE_USER', 'default-user')
        self.sources = sources
        self.ontology = ontology
        self.config = config
        self.prompt_type = prompt_type
        self.language = language
        self.config_file_path = config_file_path
        self.use_ontology = bool(use_ontology)
        self.skipped_files = []  # Track files that failed processing
        if config is None:
            self.config = {
                "max_workers": 16,
                "max_input_tokens": 100000,
                "max_output_tokens": 80000,
            }
        else:
            self.config = config
        self.model = model
        self.graph = graph
        self.hide_progress = hide_progress
        self.process_files = 0
        self.counter_lock = Lock()
        #Counters for unknown labels encountered during validation
        self.unknown_entity_labels: Counter[str] = Counter()
        self.unknown_relation_labels: Counter[str] = Counter()
        # store accepted triples with confidence for evaluation report
        self.accepted_triples: list[str] = []
        if not os.path.exists("logs"):
            os.makedirs("logs")
        # Ensure evaluation directory exists for reports
        if not os.path.exists("evaluation"):
            os.makedirs("evaluation")
        
        # Get the selected prompt pair
        try:
            prompt_pair = PromptSelector.get_prompt_pair(self.prompt_type)
            self.extract_data_system = prompt_pair.system
            self.extract_data_prompt = prompt_pair.user
            logger.info(f"✅ Using {self.prompt_type} prompts - System: {len(self.extract_data_system)} chars, User: {len(self.extract_data_prompt)} chars")
            logger.info( f" System: {self.extract_data_system}")
        except ValueError as e:
            logger.warning(f"Invalid prompt type '{self.prompt_type}': {e}. Falling back to V1.")
            prompt_pair = PromptSelector.get_prompt_pair("V1")
            self.extract_data_system = prompt_pair.system
            self.extract_data_prompt = prompt_pair.user
            logger.info(f"✅ Fallback to V1 prompts - System: {len(self.extract_data_system)} chars, User: {len(self.extract_data_prompt)} chars")

    @observe(name="CreateChatExtractDataKG",  as_type="generation")
    def _create_chat(self, documentid=None) -> GenerativeModelChatSession:
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id)
        langfuse.update_current_generation(input={"document_id": documentid}) 
        # Use minimal/empty ontology in schema-free mode to avoid biasing the system prompt
        ontology_str = str(self.ontology.to_json()) if self.use_ontology else '{"entities":[],"relations":[]}'
        return self.model.start_chat(self.extract_data_system.replace("#ONTOLOGY", ontology_str).format(language=self.language))

    @observe(name="RunDataExtractionKG")
    def run(self, instructions: Optional[str] = None):
        """
        Run the data extraction process.
        
        Args:
            instructions (Optional[str]): Optional additional instructions for data extraction.
        """
        # Each task is represented by a tuple containing:
        #   1. A Future object (the asynchronous processing task)
        #   2. A string (the ID of the document being processed)
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id) 
        tasks: list[tuple[Future, str]] = []
        
        # Collect documents from all sources
        documents = [
            document
            for source in self.sources
            for document in source.load()
            if document.not_empty()
            ]
        
        with tqdm(total=len(documents), desc="Process Documents", disable=self.hide_progress) as pbar:
            with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
                
                # Concurrency document processing
                for document in documents:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    task_id = f"extract_data_step_{timestamp}_{uuid4()}"
                    task = executor.submit(
                        self._process_document,
                        task_id,
                        self._create_chat(documentid=document),
                        document,
                        self.ontology,
                        self.graph,
                        instructions,
                    )
                    tasks.append((task, document.id))
                    
                # Wait for all tasks to be completed
                while any(task[0].running() or not task[0].done() for task in tasks):
                    time.sleep(RENDER_STEP_SIZE)
                    with self.counter_lock:
                        pbar.n = self.process_files
                    pbar.refresh()

        # Collect failed documents
        failed_documents = [task[1] for task in tasks if task[0].exception()]

        # After processing all documents, write evaluation report
        self._write_unknown_labels_report(top_n=50)
        self._write_triples_report()
        
        # Update config file with skipped files if config path is provided
        if self.config_file_path and self.skipped_files:
            self._update_config_with_skipped_files()
        
        return failed_documents

    def _validate_extracted_entity(self, entity_data: dict, ontology: Ontology, task_logger: logging.Logger) -> bool:
        """Validate extracted entity data against the ontology."""
        label = entity_data.get("label")
        attributes = entity_data.get("attributes", {})

        if not label:
            task_logger.warning(f"Skipping entity creation: Missing 'label' in {entity_data}")
            return False

        # In schema-free mode, skip ontology lookups entirely
        if self.use_ontology:
            ontology_entity = ontology.get_entity_with_label(label)
            if not ontology_entity:
                # Allow new entity labels as per prompt instructions
                task_logger.info(f"Creating new entity type '{label}': Not found in ontology but allowed per prompt rules.")
                # Track unknown entity label for reporting
                self.unknown_entity_labels.update([label])
                # Don't return False - allow creation of new entity types

        # Attribute validation skipped per new requirements.
        return True

    def _validate_extracted_relation(self, relation_data: dict, ontology: Ontology, task_logger: logging.Logger) -> bool:
        """Validate extracted relation data against the ontology."""
        rel_label = relation_data.get("label")
        source_data = relation_data.get("source")
        target_data = relation_data.get("target")
        attributes = relation_data.get("attributes", {})

        if not all([rel_label, source_data, target_data]):
            task_logger.warning(f"Skipping relation creation: Missing 'label', 'source', or 'target' in {relation_data}")
            return False

        # Skip normalization for source and target labels
        source_label = source_data.get("label")
        target_label = target_data.get("label")

        if not source_label or not target_label:
            task_logger.warning(f"Skipping relation '{rel_label}': Missing source/target label in {relation_data}")
            return False

        # Check if source and target have required attributes
        if not source_data.get("attributes", {}).get("name"):
            task_logger.warning(f"Skipping relation '{rel_label}': Missing 'name' in source attributes.")
            return False
            
        if not target_data.get("attributes", {}).get("name"):
            task_logger.warning(f"Skipping relation '{rel_label}': Missing 'name' in target attributes.")
            return False

        # In schema-free mode, skip ontology relation checks
        if self.use_ontology:
            ontology_relation_defs = ontology.get_relations_with_label(rel_label)
            if not ontology_relation_defs:
                task_logger.info(f"Creating new relation type '{rel_label}': Not found in ontology but allowed per prompt rules.")
                # Track unknown relation label for reporting
                self.unknown_relation_labels.update([rel_label])
                # Don't return False - allow creation of new relation types

        # Attribute validation skipped per new requirements.
        return True

    def _split_identifying_attributes(self, label: str, attrs: dict, ontology: Ontology) -> tuple[dict, dict]:
        """Return (identifying, payload) attribute dicts.

        identifying  – subset of attrs whose *name* is flagged ``unique=True`` in the
                      ontology for the given label.  If none of those attributes are
                      present in *attrs* we prefer using only the `name` attribute if
                      available (common stable key). If even that is missing, we fall
                      back to the full *attrs* dict so that the node can still be
                      found/created.
        payload      – attributes that will be SET after the MERGE.
        """
        # In schema-free mode, prefer 'name' as identifier when present
        if not self.use_ontology:
            if "name" in attrs and isinstance(attrs.get("name"), str) and attrs.get("name"):
                return {"name": attrs["name"]}, {k: v for k, v in attrs.items() if k != "name"}
            return attrs, {}

        entity_def = ontology.get_entity_with_label(label)
        if entity_def is None:
            return attrs, {}

        unique_names = {a.name for a in entity_def.attributes if a.unique}
        identifying = {k: v for k, v in attrs.items() if k in unique_names}

        if not identifying:
            # Prefer stable 'name' as identifier when unique flags are absent
            if "name" in attrs and isinstance(attrs.get("name"), str) and attrs.get("name"):
                return {"name": attrs["name"]}, {k: v for k, v in attrs.items() if k != "name"}
            # If extractor did not supply a unique key and no 'name' present, use everything
            # to avoid collapsing distinct nodes into one placeholder.
            return attrs, {}

        payload = {k: v for k, v in attrs.items() if k not in identifying}
        return identifying, payload

    @observe(name="ProcessDocumentExtractDataKG", as_type="generation")
    def _process_document(
        self,
        task_id: str,
        chat_session: GenerativeModelChatSession,
        document: Document,
        ontology: Ontology,
        graph: Graph,
        instructions: Optional[str] = "",
        retries: Optional[int] = 3,
    ):
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id) 
        try:
            """
            Process a single source document and extract entities and relations.
            
            Args:
                task_id (str): The unique ID for the task.
                chat_session (GenerativeModelChatSession): The chat session for the extraction.
                document (Document): The document to process.
                ontology (Ontology): The ontology associated with the graph.
                graph (Graph): The FalkorDB graph instance.
                instructions (Optional[str]): Global instructions for extraction.
                retries (Optional[int]): Number of times to retry if the model stops unexpectedly.
            """
            _task_logger = logging.getLogger(task_id)
            _task_logger.setLevel(logging.INFO)
            _task_logger.propagate = False

            if not os.path.exists("logs"):
                os.makedirs("logs")
                
            fh = logging.FileHandler(f"logs/{task_id}.log")
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
            )
            fh.setLevel(logging.INFO)
            
            _task_logger.addFilter(CypherQueryFilter())
            _task_logger.addHandler(fh)

            logger.info(f"Processing task: {task_id}")
            _task_logger.info(f"Processing task: {task_id}")
                            
            text = document.content[: self.config["max_input_tokens"]]

         
            # Build ontology section for prompt (use minimal when schema-free)
            if self.use_ontology:
                ontology_prompt = str(ontology.to_json())
            else:
                ontology_prompt = '{"entities":[],"relations":[]}'

            user_message = self.extract_data_prompt.format(
                    text=text,
                    instructions=instructions if instructions is not None else "",
                    max_tokens=self.config["max_output_tokens"],
                    ontology=ontology_prompt,
                    language=self.language,
            )

            _task_logger.info("Processing document with ID: " + document.id)

            responses: list[GenerationResponse] = []
            response_idx = 0

            responses.append(self._call_model(chat_session, user_message))

            _task_logger.info(f"Received model response #{response_idx+1}")

            while responses[response_idx].finish_reason == FinishReason.MAX_TOKENS and response_idx < retries:
                _task_logger.info("Asking model to continue")
                response_idx += 1
                responses.append(self._call_model(chat_session, COMPLETE_DATA_EXTRACTION))
                _task_logger.info(f"Received continued model response #{response_idx+1}")

            if responses[response_idx].finish_reason != FinishReason.STOP:
                _task_logger.warning(
                    f"Model stopped unexpectedly: {responses[response_idx].finish_reason}"
                )
                # Instead of raising Exception, log and return to avoid failing the whole batch
                _task_logger.error(f"Document {document.id} processing failed due to model stopping unexpectedly.")
                return # Exit processing for this document

            # Full json response is in the last response
            last_respond = responses[-1].text
            ndjson_lines = []
            try:
                ndjson_lines = parse_ndjson(last_respond)
                if not ndjson_lines:
                    _task_logger.warning(f"Could not extract NDJSON structure from response: {last_respond[:100]}...")
            except Exception as e:
                _task_logger.warning(f"Error parsing initial NDJSON: {e}. Response: {last_respond[:100]}...")
                _task_logger.info(f"Prompting model to fix NDJSON")
                try:
                    json_fix_response = self._call_model(
                        self._create_chat(),
                        FIX_JSON_PROMPT.format(json=last_respond, error=str(e)),
                    )
                    ndjson_lines = parse_ndjson(json_fix_response.text)
                    if ndjson_lines:
                        _task_logger.info(f"NDJSON fixed successfully")
                    else:
                        _task_logger.error(f"Could not extract NDJSON structure from fix response: {json_fix_response.text[:100]}...")
                except Exception as fix_e:
                    _task_logger.error(f"Error parsing fixed NDJSON: {fix_e}. Fix Response: {json_fix_response.text[:100]}...")

            # If ndjson_lines is empty or missing required types, log and return
            if not ndjson_lines:
                _task_logger.error(
                    f"Invalid or missing data after attempting NDJSON fix for document {document.id}. Skipping DB insertion."
                )
                # Track this file as skipped
                self.skipped_files.append({
                    'file_path': document.id,
                    'reason': 'No valid NDJSON extracted',
                    'timestamp': datetime.now().isoformat()
                })
                return # Exit processing for this document

            # Separate entities and relations from NDJSON lines
            entities = [obj for obj in ndjson_lines if obj.get("type") == "entity"]
            relations = [obj for obj in ndjson_lines if obj.get("type") == "relation"]
            # Log output to Langfuse
            langfuse.update_current_generation(output={"entities": entities, "relations": relations})
            # Log output to Langfuse
            langfuse.update_current_generation(output={"entities": entities, "relations": relations})
            _task_logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations. Processing...")

            # Process and create entities
            created_entities = 0
            failed_entities = 0
            for entity_data in entities:
                try:
                    is_valid_entity = self._validate_extracted_entity(entity_data, ontology, _task_logger)
                except Exception as e:
                    _task_logger.error(f"Validation error for entity: {entity_data}. Error: {e}")
                    failed_entities += 1
                    continue
                if is_valid_entity:
                    try:
                        self._create_entity(graph, entity_data, ontology)
                        created_entities += 1
                    except Exception as e:
                        # _create_entity logs its own errors, just log context here
                        _task_logger.error(f"DB error creating entity: {entity_data}. Error: {e}")
                        failed_entities += 1
                else:
                    failed_entities += 1
            _task_logger.info(f"Entity processing complete: {created_entities} created, {failed_entities} failed out of {len(entities)} total.")

            # Process and create relations
            created_relations = 0
            failed_relations = 0
            for relation_data in relations:
                try:
                    is_valid_relation = self._validate_extracted_relation(relation_data, ontology, _task_logger)
                except Exception as e:
                    _task_logger.error(f"Validation error for relation: {relation_data}. Error: {e}")
                    failed_relations += 1
                    continue
                if is_valid_relation:
                    try:
                        # Inject source document name into relation attributes
                        if "attributes" not in relation_data or not isinstance(relation_data["attributes"], dict):
                            relation_data["attributes"] = {}
                        relation_data["attributes"]["source_document"] = document.id
                        self._create_relation(graph, relation_data, ontology)
                        created_relations += 1
                    except Exception as e:
                        # _create_relation logs its own errors, just log context here
                        _task_logger.error(f"DB error creating relation: {relation_data}. Error: {e}")
                        failed_relations += 1
                else:
                    failed_relations += 1
            _task_logger.info(f"Relation processing complete: {created_relations} created, {failed_relations} failed out of {len(relations)} total.")
            
        except Exception as e:
            # Catch any unexpected errors during the process
            logger.exception(f"Task id: {task_id} failed unexpectedly - {e}")
            # Track this file as skipped due to exception
            self.skipped_files.append({
                'file_path': document.id,
                'reason': f'Processing exception: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
            # We don't re-raise here to allow other documents in the batch to process
        finally:
            with self.counter_lock:
                self.process_files += 1

    def _create_entity(self, graph: Graph, args: dict, ontology: Ontology) -> None:
        """
        Create (or update) an entity in the graph.
        Uses ontology-defined *unique* attributes as composite key for MERGE so
        that distinct real-world objects don't collapse into one generic node.
        """
        import json
        try:
            label = args.get("label")
            if not label:
                raise ValueError("Missing label in entity arguments")

            all_attrs = args.get("attributes", {})
            id_attrs, payload_attrs = self._split_identifying_attributes(label, all_attrs, ontology)

            # Include confidence as a property if provided
            confidence_val = args.get("confidence")

            merge_props = map_dict_to_cypher_properties(id_attrs)
            cypher_parts = [f"MERGE (e:`{label}` {merge_props})"]

            if confidence_val is not None:
                payload_attrs = {**payload_attrs, "confidence": confidence_val}

            if payload_attrs:
                # Separate description so we can concatenate instead of overwrite
                desc_val = payload_attrs.pop("description", None)

                # Build SET clause only for non-identifier, non-description attributes
                # Escape property names with special characters using backticks
                set_fragments = []
                for k, v in payload_attrs.items():
                    escaped_key = f"`{k}`" if any(c in k for c in "äöüßÄÖÜ") else k
                    set_fragments.append(f"e.{escaped_key} = {json.dumps(v, ensure_ascii=False)}")
                if set_fragments:
                    cypher_parts.append("SET " + ", ".join(set_fragments))

                # For description, concatenate new content to existing if not duplicate/substring
                if desc_val is not None:
                    nd = json.dumps(desc_val, ensure_ascii=False)
                    desc_set = (
                        "SET e.description = CASE "
                        f"WHEN {nd} = '' THEN e.description "
                        "WHEN e.description IS NULL OR e.description = '' THEN " + nd + " "
                        f"WHEN e.description CONTAINS {nd} THEN e.description "
                        f"WHEN {nd} CONTAINS e.description THEN {nd} "
                        "ELSE e.description + '\\n' + " + nd + " "
                        "END"
                    )
                    cypher_parts.append(desc_set)

            cypher = "\n".join(cypher_parts)
            graph.query(cypher)
        except Exception as e:
            logger.error(f"Error creating entity for {args}: {e}")
            raise

    def _create_relation(self, graph: Graph, args: dict, ontology: Ontology) -> None:
        """
        Create (or update) a relationship in the graph.  Source and target nodes
        are matched using their identifying (unique) attributes as defined in
        the ontology.
        """
        import json
        try:
            source_type = args.get("source", {}).get("label")
            target_type = args.get("target", {}).get("label")
            rel_type    = args.get("label")
            if not (source_type and target_type and rel_type):
                logger.error(f"Incomplete relation data: {args}")
                return

            # Keep relation type as provided (no upper-case normalization)
            rel_type = rel_type

            source_attrs_full = args.get("source", {}).get("attributes", {})
            target_attrs_full = args.get("target", {}).get("attributes", {})
            rel_attrs         = args.get("attributes", {})

            # Persist confidence score as a property if present
            confidence_val = args.get("confidence")
            if confidence_val is not None:
                rel_attrs = {**rel_attrs, "confidence": confidence_val}

            # Skip normalization for source and target labels
            source_label = source_type
            target_label = target_type

            src_id_attrs, _ = self._split_identifying_attributes(source_label, source_attrs_full, ontology)
            tgt_id_attrs, _ = self._split_identifying_attributes(target_label, target_attrs_full, ontology)

            src_props = map_dict_to_cypher_properties(src_id_attrs)
            tgt_props = map_dict_to_cypher_properties(tgt_id_attrs)
            rel_props = map_dict_to_cypher_properties(rel_attrs)

            cypher = f"""
            MERGE (s:`{source_label}` {src_props})
            MERGE (t:`{target_label}` {tgt_props})
            MERGE (s)-[r:`{rel_type}` {rel_props}]->(t)
            """
            graph.query(cypher)

            # Record triple for evaluation report with source document
            source_doc = args.get('source_document', 'unknown_document')
            triple_str = f"({source_type})-[:{rel_type}]->({target_type})\tconfidence:{confidence_val if confidence_val is not None else 'N/A'}\tsource:{source_doc}"
            self.accepted_triples.append(triple_str)
        except Exception as e:
            logger.error(f"Error creating relation for {args}: {e}")
            raise

    def _update_config_with_skipped_files(self) -> None:
        """
        Update the config file to include skipped files for retry processing.
        This method adds skipped files to the sources list in the config file.
        """
        if not self.config_file_path or not os.path.exists(self.config_file_path):
            logger.warning(f"Config file not found: {self.config_file_path}")
            return
        
        try:
            # Load existing config
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Get current sources list
            current_sources = config_data.get('sources', [])
            
            # Extract file paths from skipped files
            skipped_file_paths = [skipped['file_path'] for skipped in self.skipped_files]
            
            # Add skipped files to sources if they're not already there
            new_sources = []
            for skipped_path in skipped_file_paths:
                if skipped_path not in current_sources:
                    new_sources.append(skipped_path)
                    logger.info(f"Adding skipped file to config for retry: {skipped_path}")
            
            if new_sources:
                # Update sources list
                config_data['sources'].extend(new_sources)
                
                # Add metadata about skipped files
                config_data['skipped_files'] = self.skipped_files
                config_data['last_updated'] = datetime.now().isoformat()
                
                # Save updated config
                with open(self.config_file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Updated config file with {len(new_sources)} skipped files for retry")
            else:
                logger.info("No new skipped files to add to config")
                
        except Exception as e:
            logger.error(f"Error updating config file with skipped files: {e}")

    @sleep_and_retry
    @limits(calls=15, period=60)
    def _call_model(
        self,
        chat_session: GenerativeModelChatSession,
        prompt: str,
        retry: int = 6,
        output_method: OutputMethod = OutputMethod.DEFAULT
    ) -> GenerationResponse:
        """
        Call the generative model with rate limiting and retries.
        
        Args:
            chat_session (GenerativeModelChatSession): The chat session for interacting with the model.
            prompt (str): The prompt to send to the model.
            retry (Optional[int]): Number of retries in case of quota exceeded or errors.
            output_method (Optional[OutputMethod]): The output method for the response.
        
        Returns:
            GenerationResponse: The model's response.
        
        Raises:
            Exception: If an error occurs after exhausting retries.
        """
        try:
            return chat_session.send_message(prompt, output_method=output_method)
        except Exception as e:
            # If exception is caused by quota exceeded, wait 10 seconds and try again for 6 times
            if "Quota exceeded" in str(e) and retry > 0:
                time.sleep(10)
                retry -= 1
                return self._call_model(chat_session, prompt, retry)
            else:
                if retry == 0:
                    logger.error("Quota exceeded")
                raise e

    # NEW ------------------------------------------------------------------
    def _write_unknown_labels_report(self, top_n: int = 50):
        """Write a report of the most frequent unknown labels encountered."""
        if not self.unknown_entity_labels and not self.unknown_relation_labels:
            return  # nothing to write

        report_path = os.path.join("evaluation", "unknown_labels_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as fp:
                fp.write("=== Unknown Entity Labels ===\n")
                for label, count in self.unknown_entity_labels.most_common(top_n):
                    fp.write(f"{label}: {count}\n")

                fp.write("\n=== Unknown Relation Labels ===\n")
                for label, count in self.unknown_relation_labels.most_common(top_n):
                    fp.write(f"{label}: {count}\n")

            logger.info(f"Unknown-labels report written to {report_path}")
        except Exception as e:
            logger.error(f"Failed to write unknown-labels report: {e}")

    def _write_triples_report(self):
        """Write all accepted relation triples with their confidence score."""
        if not self.accepted_triples:
            return

        report_path = Path("evaluation") / "accepted_triples_with_confidence.txt"
        try:
            with report_path.open("w", encoding="utf-8") as fp:
                for line in self.accepted_triples:
                    fp.write(line + "\n")
            logger.info(f"Triples report written to {report_path}")
        except Exception as e:
            logger.error(f"Failed to write triples report: {e}")
