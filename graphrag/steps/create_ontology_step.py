import time
import json
import logging
import os
from tqdm import tqdm
from threading import Lock
from typing import Optional, Dict, List
from ..steps.Step import Step
from ..document import Document
from ..ontology import Ontology
from ..helpers import parse_ndjson, extract_json
from ratelimit import limits, sleep_and_retry
from ..source import AbstractSource
from concurrent.futures import Future, ThreadPoolExecutor
import spacy
from collections import defaultdict
from ..models import (
    GenerativeModel,
    GenerativeModelChatSession,
    GenerationResponse,
    FinishReason,
)
from datetime import datetime
from ..fixtures.prompts import (
    CREATE_ONTOLOGY_SYSTEM,
    CREATE_ONTOLOGY_PROMPT,
    FIX_ONTOLOGY_PROMPT,
    FIX_JSON_PROMPT,
    BOUNDARIES_PREFIX,
)
from ..label_normalizer import LabelNormalizer

from langfuse import observe, get_client
langfuse = get_client()  # Initialize Langfuse client for tracing
# set session_id in format DDMMYY-RAG-Generation
session_id =  f"default-session-{datetime.now().strftime('%Y%m%d')}"
langfuse_user_id = "default-user"  # Default user ID for Langfuse

RENDER_STEP_SIZE = 0.5
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CreateOntologyStep(Step):
    """
    Create Ontology Step with enhanced Named Entity Recognition using spaCy
    """

    def __init__(
        self,
        sources: list[AbstractSource],
        ontology: Ontology,
        model: GenerativeModel,
        config: Optional[dict] = None,
        hide_progress: Optional[bool] = False,
        language: Optional[str] = "GERMAN",
    ) -> None:
        """
        Initialize the CreateOntologyStep.
        
        Args:
            sources (List[AbstractSource]): List of sources from which ontology is created.
            ontology (Ontology): The initial ontology to be merged with new extracted data.
            model (GenerativeModel): The generative model used for processing and creating the ontology.
            config (Optional[dict]): Configuration for the step, including thread workers and token limits. Defaults to standard config.
            hide_progress (bool): Flag to hide the progress bar. Defaults to False.
            language (Optional[str]): Language name for prompts ("German" or "English"). Defaults to "GERMAN".
        """
        global session_id, langfuse_user_id
        session_id = os.getenv('GRAPHRAG_SESSION_ID', session_id)  #
        langfuse_user_id = os.getenv('LANGFUSE_USER', 'default-user')
        self.sources = sources
        self.ontology = ontology
        self.model = model
        self.language = language
        if config is None:
            self.config = {
                "max_workers": 16,
                "max_input_tokens": 100000,
                "max_output_tokens": 80000,
                "spacy_model": "de_core_news_sm",  # Options: "de_core_news_sm", "en_core_web_md", "de_core_news_sm", "en_core_web_trf"
            }
        else:
            self.config = config
            # Ensure spacy_model is set
            if "spacy_model" not in self.config:
                self.config["spacy_model"] = "de_core_news_sm"
                
        self.hide_progress = hide_progress
        self.process_files = 0
        self.counter_lock = Lock()
        
        # Initialize spaCy - use lazy loading to avoid loading the model unless needed
        self.nlp = None
        
        # Initialize label normalizer
        self.normalizer = LabelNormalizer()
        
    def _load_spacy_model(self):
        """
        Lazy-load the spaCy model on first use.
        """
        if self.nlp is None:
            try:
                self.nlp = spacy.load(self.config["spacy_model"])
                logger.info(f"Successfully loaded spaCy model: {self.config['spacy_model']}")
            except OSError as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                logger.warning(f"To use spaCy features, please install the model with: python -m spacy download {self.config['spacy_model']}")
                # Fall back to a blank model that will just tokenize 
                self.nlp = spacy.blank("en")
        return self.nlp

    @observe(name="NER_with_Spacy_Ontology", as_type="generation")      
    def _extract_entities_with_spacy(self, text: str) -> str:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text (str): The text to extract entities from.
            
        Returns:
            str: A formatted string of entities for the LLM prompt.
        """
        if not text:
            return ""
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id)    
        nlp = self._load_spacy_model()
        
        # Process text with spaCy, handling potential errors
        try:
            # Limit text size to avoid memory issues (adjust as needed)
            max_chars = min(len(text), 100000)  # 100K chars max to avoid memory issues
            doc = nlp(text[:max_chars])
            
            # Extract named entities
            entities_by_type = defaultdict(list)
            for ent in doc.ents:
                # Store unique entities (case-insensitive)
                if ent.text.lower() not in [e.lower() for e in entities_by_type[ent.label_]]:
                    entities_by_type[ent.label_].append(ent.text)
            
            # Also extract noun chunks as potential entities
            noun_chunks = []
            for chunk in doc.noun_chunks:
                # Filter out pronouns and very short chunks
                if (len(chunk.text) > 3 and 
                    not chunk.text.lower() in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those']):
                    # Store unique chunks (case-insensitive)
                    if chunk.text.lower() not in [nc.lower() for nc in noun_chunks]:
                        noun_chunks.append(chunk.text)
            
            # Format the results
            result = "**Preliminary Entities Identified by Named Entity Recognition:**\n\n"
            
            # Add named entities grouped by type
            if entities_by_type:
                result += "**Named Entities by Type:**\n"
                for entity_type, entities in entities_by_type.items():
                    if entities:  # Only include non-empty entity types
                        result += f"- **{entity_type}**: {', '.join(entities[:20])}"
                        if len(entities) > 20:
                            result += f" and {len(entities) - 20} more"
                        result += "\n"
                result += "\n"
            
            # Add noun chunks if available
            if noun_chunks:
                result += "**Additional Potential Entities (Noun Phrases):**\n"
                # Limit to first 50 chunks to avoid overwhelming the prompt
                result += f"- {', '.join(noun_chunks[:50])}"
                if len(noun_chunks) > 50:
                    result += f" and {len(noun_chunks) - 50} more"
                result += "\n\n"
            
            logger.debug(f"Extracted {sum(len(entities) for entities in entities_by_type.values())} named entities and {len(noun_chunks)} noun chunks using spaCy.")
            return result
            
        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")
            langfuse.update_current_generation(output=f"Error in spaCy entity extraction: {e}")
            return ""

    @observe(name="CreateChatOntology", as_type="generation")
    def _create_chat(self) -> GenerativeModelChatSession:
        """
        Create a new chat session with the generative model.
        
        Returns:
            GenerativeModelChatSession: A session for interacting with the generative model.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id) 
        return self.model.start_chat(CREATE_ONTOLOGY_SYSTEM.format(language=self.language, max_tokens=""))

    @observe(name="RunOntology",  as_type="generation")
    def run(self, boundaries: Optional[str] = None):
        """
        Execute the ontology creation process by extracting data from all sources.
        
        Args:
            boundaries (Optional[str]): Additional boundaries or constraints for the ontology creation.
            
        Returns:
            Ontology: The final ontology after merging with extracted data.
        
        Raises:
            Exception: If ontology creation fails and no entities are found.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id) 
        tasks: list[Future[Ontology]] = []

        with tqdm(total=len(self.sources) + 1, desc="Process Documents", disable=self.hide_progress) as pbar:
            with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:

                # Process each source document in parallel
                for source in self.sources:
                    task = executor.submit(
                        self._process_source,
                        self._create_chat(),
                        source,
                        self.ontology,
                        boundaries,
                    )
                    tasks.append(task)

                # Wait for all tasks to be completed
                while any(task.running() or not task.done() for task in tasks):
                    time.sleep(RENDER_STEP_SIZE)
                    with self.counter_lock:
                        pbar.n = self.process_files
                    pbar.refresh()
                
                # Validate the ontology
                if len(self.ontology.entities) == 0:
                    raise Exception("Failed to create ontology")
                
                # Finalize the ontology
                task_fin = executor.submit(self._fix_ontology, self._create_chat(), self.ontology)
                
                # Wait for the final task to be completed
                while not task_fin.done():
                    time.sleep(RENDER_STEP_SIZE)
                    pbar.refresh()
                pbar.update(1)

        # POST-ONTOLOGY CREATION NORMALIZATION
        logger.info("🔧 Normalizing ontology with hierarchical typing...")
        self.ontology = self._normalize_ontology_post_creation()
        logger.info(f"✅ Normalization completed - {len(self.ontology.entities)} entities, {len(self.ontology.relations)} relations")

        return self.ontology

    @observe(name="ProcessSourceOntology", as_type="generation")
    def _process_source(
        self,
        chat_session: GenerativeModelChatSession,
        source: AbstractSource,
        o: Ontology,
        boundaries: Optional[str] = None,
        retries: Optional[int] = 3,
    ):
        """
        Process a single document and extract ontology data.
        
        Args:
            chat_session (GenerativeModelChatSession): The chat session for interacting with the generative model.
            document (Document): The document to extract data from.
            o (Ontology): The current ontology to be merged with extracted data.
            boundaries (Optional[str]): Constraints for data extraction.
            retries (Optional[int]) Number of retries for processing the document.
            
        Returns:
            Ontology: The updated ontology after processing the document.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id) 
        try:
            document = next(source.load())
            
            text = document.content[: self.config["max_input_tokens"]]
            
            # Terminal notification for spaCy processing
            print(f"\n🚀 Processing document with spaCy: {document.id}")
            print(f"📄 Document length: {len(document.content)} chars (processing first {len(text)} chars)")
            
            # Extract named entities from the text using spaCy
            spacy_ner_results = self._extract_entities_with_spacy(text)
            logger.debug(f"spaCy found entities in document: {document.id}")

            user_message = CREATE_ONTOLOGY_PROMPT.format(
                text=text,
                boundaries=BOUNDARIES_PREFIX.format(user_boundaries=boundaries) if boundaries is not None else "",
                spacy_ner_results=spacy_ner_results,
                max_tokens=self.config["max_output_tokens"],
                language=self.language
            )

            responses: list[GenerationResponse] = []
            response_idx = 0

            responses.append(self._call_model(chat_session, user_message))

            logger.debug(f"Model response: {responses[response_idx]}")

            while responses[response_idx].finish_reason == FinishReason.MAX_TOKENS and response_idx < retries:
                response_idx += 1
                responses.append(self._call_model(chat_session, "continue"))

            if responses[response_idx].finish_reason != FinishReason.STOP:
                raise Exception(
                    f"Model stopped unexpectedly: {responses[response_idx].finish_reason}"
                )

            combined_text = " ".join([r.text for r in responses])
            logger.info(f"Combined text (first 200 chars): {combined_text[:200]}")


            try:
                data = parse_ndjson(combined_text)
            except Exception as e:
                logger.debug(f"Error extracting NDJSON: {e}")
                logger.debug(f"Prompting model to fix NDJSON")
                json_fix_response = self._call_model(
                    self._create_chat(),
                    FIX_JSON_PROMPT.format(json=combined_text, error=str(e)),
                )
                try:
                    data = parse_ndjson(json_fix_response.text)
                    logger.debug(f"Fixed NDJSON: {data}")
                except Exception as e:
                    logger.info(f"MISTAKE IN PROCESS SOURCE ")
                    logger.error(f"Failed to fix NDJSON: {e}  {json_fix_response.text}")
                    data = None

            if data is not None and len(data) > 0:
                try:
                    new_ontology = Ontology.from_ndjson(data)
                except Exception as e:
                    logger.error(f"Exception while extracting NDJSON: {e}\nModel output was: {combined_text[:500]}")
                    new_ontology = None
            else:
                logger.warning(f"No valid NDJSON extracted from model for document {document.id}. Model output was: {combined_text[:500]}")
                new_ontology = None

            if new_ontology is not None:
                o = o.merge_with(new_ontology)

            logger.debug(f"Processed document: {document}")
        except Exception as e:
            logger.exception(f"Failed - {e}")
            raise e
        finally:
            with self.counter_lock:
                self.process_files += 1
            return o
        
    @observe(name="FixOntology", as_type="generation")
    def _fix_ontology(self, chat_session: GenerativeModelChatSession, o: Ontology):
        """
        Fix and validate the ontology using the generative model.
        
        Args:
            chat_session (GenerativeModelChatSession): The chat session for interacting with the model.
            o (Ontology): The ontology to fix and validate.
            
        Returns:
            Ontology: The fixed and validated ontology.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id) 
        logger.debug(f"Fixing ontology...")

        user_message = FIX_ONTOLOGY_PROMPT.format(ontology=o)

        responses: list[GenerationResponse] = []
        response_idx = 0

        responses.append(self._call_model(chat_session, user_message))

        logger.debug(f"Model response: {responses[response_idx]}")

        while responses[response_idx].finish_reason == FinishReason.MAX_TOKENS:
            response_idx += 1
            responses.append(self._call_model(chat_session, "continue"))

        if responses[response_idx].finish_reason != FinishReason.STOP:
            raise Exception(
                f"Model stopped unexpectedly: {responses[response_idx].finish_reason}"
            )

        combined_text = " ".join([r.text for r in responses])

        try:
            data = parse_ndjson(combined_text)
        except Exception as e:
            logger.debug(f"Error extracting NDJSON: {e}")
            logger.debug(f"Prompting model to fix NDJSON")
            json_fix_response = self._call_model(
                self._create_chat(),
                FIX_JSON_PROMPT.format(json=combined_text, error=str(e)),
            )
            try:
                data = parse_ndjson(json_fix_response.text)
                logger.debug(f"Fixed NDJSON: {data}")
            except Exception as e:
                logger.info(f"MISTAKE IN Fix ontology ")
                logger.error(f"Failed to fix NDJSON: {e} {json_fix_response.text}")
                data = None

        if data is None or len(data) == 0:
            logger.warning(f"No valid NDJSON extracted during ontology fix. Model output was: {combined_text[:500]}")
            return o
        try:
            new_ontology = Ontology.from_ndjson(data)
        except Exception as e:
            logger.debug(f"Exception while extracting NDJSON: {e}\nModel output was: {combined_text[:500]}")
            new_ontology = None

        if new_ontology is not None:
            o = o.merge_with(new_ontology)

        logger.debug(f"Fixed ontology: {o}")

        return o

    @observe(name="NormalizeOntology",  as_type="generation")
    def _normalize_ontology_post_creation(self) -> Ontology:
        """
        POST-ONTOLOGY CREATION NORMALIZATION
        Simplified normalization for German data - basic cleaning, abbreviation expansion, and title case.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id) 
        from ..entity import Entity
        from ..relation import Relation
        
        normalized_entities = []
        normalized_relations = []
        
        # Log the Input to Langfuse
        langfuse.update_current_generation(
            input={
                "entities": self.ontology.entities,
                "relations": self.ontology.relations
            }
        )
        logger.info(f"Starting German normalization of {len(self.ontology.entities)} entities and {len(self.ontology.relations)} relations")
        
        # Step 1: Normalize entities
        for entity in self.ontology.entities:
            normalized_label, _ = self.normalizer.normalize_entity_label(
                entity.label, 
                entity.description
            )
            
            # Create normalized entity
            normalized_entity = Entity(
                label=normalized_label,
                attributes=entity.attributes,
                description=entity.description
            )
            normalized_entities.append(normalized_entity)
        
        # Step 2: Normalize relations
        for relation in self.ontology.relations:
            normalized_label = self.normalizer.normalize_relation_label(relation.label)
            
            # Normalize source/target labels
            source_normalized, _ = self.normalizer.normalize_entity_label(relation.source.label)
            target_normalized, _ = self.normalizer.normalize_entity_label(relation.target.label)
            
            normalized_relation = Relation(
                label=normalized_label,
                source=source_normalized,
                target=target_normalized,
                attributes=relation.attributes
            )
            normalized_relations.append(normalized_relation)
        
        logger.info(f"German normalization complete: {len(normalized_entities)} entities, {len(normalized_relations)} relations")
        
        return Ontology(normalized_entities, normalized_relations)

    @sleep_and_retry
    @limits(calls=15, period=60)
    def _call_model(
        self,
        chat_session: GenerativeModelChatSession,
        prompt: str,
        retry: Optional[int] = 6,
    ):
        """
        Call the generative model with retries and rate limiting.
        
        Args:
            chat_session (GenerativeModelChatSession): The chat session for interacting with the model.
            prompt (str): The prompt to send to the model.
            retry (Optional[int]): Number of retries if quota is exceeded or errors occur.
            
        Returns:
            GenerationResponse: The model's response.
            
        Raises:
            Exception: If the model fails after exhausting retries.
        """
        try:
            return chat_session.send_message(prompt)
        except Exception as e:
            # If exception is caused by quota exceeded, wait 10 seconds and try again for 6 times
            if "Quota exceeded" in str(e) and retry > 0:
                time.sleep(10)
                retry -= 1
                return self._call_model(chat_session, prompt, retry)
            else:
                raise e

    