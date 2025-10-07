"""
Enhanced GraphRAG Demo

This demo shows how to use GraphRAG for an advanced RAG setup with:
1. Define sources (docs, websites, PDFs, JSON, DOCX, HTML)
2. Automatic ontology generation from diverse data
3. Knowledge graph creation with FalkorDB
4. Vector embeddings for semantic search
5. Multi-layered responses with:
   - Level 1: Concise 4-5 word answer
   - Level 2: Detailed information
   - Level 3: Source references
6. Human verification flagging for sensitive operations
7. Alternative suggestions when exact matches aren't found
8. Multiple LLM support (Gemini, Claude)
"""

import os
import json
import time
import logging
import argparse
import traceback
from dotenv import load_dotenv
import litellm  # Import litellm
# Configure litellm to drop params that aren't supported by the model
litellm.drop_params = True
# Import components from graphrag
from graphrag.models import LiteModel
from graphrag.models.model import GenerativeModelConfig
from graphrag.source import Source
from graphrag.ontology import Ontology
from graphrag.kg import KnowledgeGraph
from graphrag.model_config import KnowledgeGraphModelConfig
from graphrag.graph_manager import GraphConfig, GraphManager
from graphrag.fixtures.prompts import (
    GRAPH_QA_SYSTEM, GRAPH_QA_PROMPT,
    MULTI_LEVEL_QA_SYSTEM, MULTI_LEVEL_QA_PROMPT
)
from graphrag.fixtures.prompts_for_KG import UPDATE_ONTOLOGY_PROMPT  # For ontology updates
from DataExtractor.main import main as data_extractor_main # Import DataExtractor main function
import sys
import io
import re
import signal
from pathlib import Path
from datetime import datetime
from redis import Redis
from falkordb import FalkorDB

LANGFUSE_USER = "Aysha_aiio" # Set your Langfuse user ID

# German text processing system instruction - defined once and reused
GERMAN_TEXT_SYSTEM_INSTRUCTION = "You are processing German text. Always preserve German vocabulary and do not translate German content to English. Output all entity names, descriptions, and attribute values in German when processing data which is in german language."

# Redis/FalkorDB connection constants for efficiency
REDIS_HOST =  "127.0.0.1"
REDIS_PORT = 6379
# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Force all loggers to error only
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger(name).propagate = False

# Add environment variables to control debug output
os.environ["GRAPHRAG_DEBUG"] = "1"
os.environ["FALKORDB_DEBUG"] = "1"
os.environ["LITELLM_DEBUG"] = "DEBUG"
os.environ["PYTHONWARNINGS"] = "default"
os.environ["TQDM_DISABLE"] = "0"

# Dummy file-like object to discard all output
class NullIO(io.StringIO):
    def write(self, txt):
        pass

# Store the original stdout and stderr
ORIGINAL_STDOUT = sys.stdout
ORIGINAL_STDERR = sys.stderr

# Flag to control verbose output
SILENT_MODE = False

def suppress_output():
    """Redirect stdout and stderr to null"""
    if SILENT_MODE:
        sys.stdout = NullIO()
        sys.stderr = NullIO()

def restore_output():
    """Restore stdout and stderr"""
    if SILENT_MODE:
        sys.stdout = ORIGINAL_STDOUT
        sys.stderr = ORIGINAL_STDERR

def print_only_essential(message):
    """Print message even in silent mode"""
    restore_output()
    print(message)
    suppress_output()

def display_structured_response(response):
    """Display a structured multi-level response in a readable format."""
    print("\n" + "=" * 70)
    
    # Level 1: Brief answer (concise 4-5 words)
    print(f"BRIEF ANSWER: {response['brief_answer']}")
    print("-" * 70)
    
    # Level 2: Detailed information
    print("DETAILED INFORMATION:")
    
    # Handle detailed_info in different formats
    if 'detailed_info' in response and response['detailed_info']:
        detailed_info = response['detailed_info']
        
        # If detailed_info is a string, display it directly
        if isinstance(detailed_info, str):
            print(detailed_info)
        else:
            # Process structured information
            for section_name, section_data in detailed_info.items():
                if section_data and isinstance(section_data, dict):
                    print(f"\n📌 {section_name.upper()}:")
                    for key, value in section_data.items():
                        if isinstance(value, list):
                            print(f"  {key.title()}:")
                            for item in value:
                                print(f"   - {item}")
                        elif value:
                            print(f"  {key.title()}: {value}")
                elif section_data and isinstance(section_data, list):
                    print(f"\n📌 {section_name.upper()}:")
                    for i, item in enumerate(section_data, 1):
                        print(f"  {i}. {item}")
    else:
        # Fallback to standard detailed information
        print(response.get('detailed', 'No detailed information available'))
    
    # Display verification warnings if needed
    if response.get('requires_verification', False):
        print("\n⚠️ " )
        print(f"REQUIRES HUMAN VERIFICATION: {response.get('verification_reason', 'Unknown reason')}")
        print("⚠️ " )
    
    # Level 3: Sources
    if response.get('sources', []):
        print("-" * 70)
        print("SOURCES:")
        for source in response['sources']:
            print(f"- {source}")
    
    # Display alternatives if there are gaps
    alternatives_value = response.get('alternatives')
    if response.get('has_gaps', False) and alternatives_value:
        print("-" * 70)
        print("ALTERNATIVES:")

        # Normalize alternatives to an iterable list of items
        if isinstance(alternatives_value, list):
            alternatives_iter = alternatives_value
        elif isinstance(alternatives_value, dict):
            alternatives_iter = [alternatives_value]
        elif isinstance(alternatives_value, (str, int, float, bool)):
            alternatives_iter = [alternatives_value]
        else:
            alternatives_iter = [str(alternatives_value)]

        for alt in alternatives_iter:
            # Accept dicts, strings, numbers, and booleans
            if isinstance(alt, dict):
                desc = (
                    alt.get('description')
                    or alt.get('suggestion')
                    or alt.get('text')
                    or alt.get('message')
                )
                # Fallback to JSON string if no known field is present
                if desc is None:
                    try:
                        desc = json.dumps(alt, ensure_ascii=False)
                    except Exception:
                        desc = str(alt)
                print(f"- {desc}")
            else:
                print(f"- {alt}")
    
    print("=" * 70)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Request timed out")


def initialize_model(model_name):
    """Initialize the LLM model based on user selection"""
    print_only_essential(f"Initializing model: {model_name}")
    
    # Create a more aggressive generation configuration with higher temperature and top_p
    # for better entity and relationship extraction
    generation_config = GenerativeModelConfig(
        temperature=0,  # Higher temperature for more creative extraction
        top_p=0.95,       # Higher top_p to consider more token options
        top_k=40,         # Consider more tokens at each step
        max_tokens= 8192,  # Reasonable limit to avoid MAX_TOKENS errors
        frequency_penalty=0.2,  # Reduce repetition 
        presence_penalty=0.2,   # Encourage more diverse extraction
    )

    print_only_essential(f"Generation config: {generation_config} and model_name: {model_name}")
    
    # Vertex AI auto-switch: if Vertex env is set, prefer vertex_ai provider
    vertex_project = os.getenv("VERTEXAI_PROJECT")
    vertex_location = os.getenv("VERTEXAI_LOCATION") or os.getenv("VERTEX_LOCATION")

    # Normalize region if user passes human-readable input (e.g., "europe west 1")
    if vertex_location and " " in vertex_location:
        vertex_location = vertex_location.replace(" ", "-")

    # If user passed bare model id without provider, add appropriate provider
    if "/" not in model_name:
        if vertex_project and vertex_location:
            model_name = f"vertex_ai/{model_name}"
        else:
            model_name = f"gemini/{model_name}"

    # Handle different model types with appropriate API key checks
    if model_name.startswith("vertex_ai/"):
        # Vertex AI uses ADC; ensure env is present
        if not (vertex_project and vertex_location):
            print("Warning: VERTEXAI_PROJECT or VERTEXAI_LOCATION not set. Falling back to Gemini API Studio if available.")
        return LiteModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=GERMAN_TEXT_SYSTEM_INSTRUCTION
        )

    if model_name.startswith("gemini"):
        if not os.getenv("GOOGLE_API_KEY"):
            print("Warning: GOOGLE_API_KEY environment variable not set. Gemini models require it.")
        return LiteModel(
            model_name=model_name, 
            generation_config=generation_config,
            system_instruction=GERMAN_TEXT_SYSTEM_INSTRUCTION
        )
    
    elif model_name.startswith("claude") or model_name.startswith("anthropic"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY environment variable not set. Claude models require it.")
        return LiteModel(
            model_name=model_name, 
            generation_config=generation_config, 
            api_key=api_key,
            system_instruction=GERMAN_TEXT_SYSTEM_INSTRUCTION
        )
    
    # Add support for OpenAI models
    elif model_name.startswith("openai"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY environment variable not set. OpenAI models require it.")
        return LiteModel(
            model_name=model_name, 
            generation_config=generation_config, 
            api_key=api_key,
            system_instruction=GERMAN_TEXT_SYSTEM_INSTRUCTION
        )
    
    # Add support for Groq models
    elif model_name.startswith("groq"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("Warning: GROQ_API_KEY environment variable not set. OpenAI models require it.")
        return LiteModel(
            model_name=model_name, 
            generation_config=generation_config, 
            api_key=api_key,
            system_instruction=GERMAN_TEXT_SYSTEM_INSTRUCTION
        )
    else:
        # Default fallback to available models
        print_only_essential(f"Model {model_name} not supported. Checking available models.")
        
        if os.getenv("GOOGLE_API_KEY"):
            return LiteModel(
                model_name="gemini/gemini-2.5-flash-preview-05-20", 
                generation_config=generation_config,
                system_instruction=GERMAN_TEXT_SYSTEM_INSTRUCTION
            )
        elif os.getenv("ANTHROPIC_API_KEY"):
            return LiteModel(
                model_name="anthropic/claude-3-sonnet-20240229", 
                generation_config=generation_config,
                system_instruction=GERMAN_TEXT_SYSTEM_INSTRUCTION
            )
        else:
            print_only_essential("No API keys found. Using Gemini as default, but it may fail without an API key.")
            return LiteModel(
                model_name="gemini/gemini-2.5-flash-preview-05-20", 
                generation_config=generation_config,
                system_instruction=GERMAN_TEXT_SYSTEM_INSTRUCTION
            )

def clear_kg_data_preserve_ontology(kg):
    """Clear only the current KG's data within its graph; ontology in memory is preserved.

    This avoids dropping other graphs by operating strictly inside kg.graph.
    """
    print("🧹 Clearing existing KG nodes/relationships (preserving in-memory ontology and other graphs)...")

    try:
        # Remove all nodes/relationships in the currently selected graph only
        try:
            kg.graph.query("MATCH (n) DETACH DELETE n")
            print("  ✓ Deleted all nodes and relationships in current graph")
        except Exception as drop_err:
            print(f"  ⚠️ Warning: Failed to delete nodes in current graph: {drop_err}")

        # Reset processed tracking; caller can reprocess as needed
        kg.processed_sources = set()

        # Refresh schema so downstream prompt injection uses the clean slate
        kg.refresh_schema()
        print("✅ Cleared graph data and refreshed schema (ontology preserved in memory)")

    except Exception as e:
        print(f"❌ Error clearing KG: {e}")
        raise

def update_ontology_via_llm(model, existing_ontology_ndjson: str, new_text: str, boundaries: str = "", language: str = "de"):
    """Use UPDATE_ONTOLOGY_PROMPT to update an existing ontology with new text.

    Returns Ontology instance on success, or None on failure.
    """
    try:
        session = model.start_chat()
        prompt = UPDATE_ONTOLOGY_PROMPT.format(
            boundaries=boundaries or "",
            ontology=existing_ontology_ndjson,
            text=new_text,
        )
        resp = session.send_message(prompt)
        resp_text = getattr(resp, "text", None) or resp  # support dict or str
        # Parse as NDJSON
        try:
            # Accept both raw string NDJSON and list of JSON lines
            from graphrag.ontology import Ontology
            updated = Ontology.from_ndjson(resp_text)
            return updated
        except Exception:
            # Try to extract JSON/NDJSON lines from text
            try:
                extracted = extract_json_from_response(resp_text)
                from graphrag.ontology import Ontology
                return Ontology.from_ndjson(extracted)
            except Exception:
                return None
    except Exception:
        return None

def extract_json_from_response(response_text):
        # """
        # Extracts JSON object from a string, even if there's surrounding text.
        
        # Args:
        #     response_text (str): The full LLM response containing JSON.

        # Returns:
        #     dict or list: Parsed JSON data.
            
        # Raises:
        #     ValueError: If no valid JSON object is found.
        # """
    # Regular expression to extract the first JSON object or array
    json_match = re.search(r'(\{.*?\}|\[.*?\])', response_text, re.DOTALL)
    
    if not json_match:
        raise ValueError("No JSON object or array found in the response.")
    
    json_str = json_match.group(1)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")

def list_knowledge_graphs(kg_manager):
    """List all existing knowledge graphs."""
    configs = kg_manager.list_graphs()
    
    if not configs:
        print("No existing knowledge graphs found.")
        return
    
    print("Existing Knowledge Graphs:")
    print("-" * 60)
    print(f"{'Name':<20} {'Created':<12} {'Nodes':<8} {'Relations':<10} {'Model':<15}")
    print("-" * 60)
    
    for config in configs:
        created = config.created_at.strftime("%Y-%m-%d")
        print(f"{config.name:<20} {created:<12} {config.node_count:<8} {config.relationship_count:<10} {config.model_name:<15}")

def delete_knowledge_graph(kg_manager, kg_name):
    """Delete a specific knowledge graph."""
    config = kg_manager.load_config(kg_name)
    if not config:
        print(f"Knowledge graph '{kg_name}' not found.")
        return
    
    # Confirm deletion
    print(f"Knowledge Graph: {kg_name}")
    print(f"Created: {config.created_at}")
    print(f"Nodes: {config.node_count}, Relations: {config.relationship_count}")
    
    confirm = input(f"\nAre you sure you want to delete '{kg_name}'? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Deletion cancelled.")
        return
    
    try:
        # Delete from database
        db = FalkorDB(host=REDIS_HOST, port=REDIS_PORT)
        graphs = db.list_graphs()
        
        if kg_name in graphs:
            graph = db.select_graph(kg_name)
            graph.delete()
            print(f"✓ Deleted graph from database: {kg_name}")
        else:
            print(f"⚠️ Graph not found in database: {kg_name}")
        
        # Delete configuration
        if kg_manager.delete_config(kg_name):
            print(f"✓ Deleted configuration: {kg_name}")
        
        print(f"Successfully deleted knowledge graph: {kg_name}")
        
    except Exception as e:
        print(f"Error deleting knowledge graph '{kg_name}': {e}")

def show_kg_info(kg_manager, kg_name):
    """Show detailed information about a specific knowledge graph."""
    config = kg_manager.load_config(kg_name)
    if not config:
        print(f"Knowledge graph '{kg_name}' not found.")
        return
    
    print(f"Knowledge Graph: {kg_name}")
    print("=" * 50)
    print(f"Created: {config.created_at}")
    print(f"Model: {config.model_name}")
    print(f"Ontology File: {config.ontology_file}")
    print(f"Sources: {len(config.sources)}")
    print(f"Nodes: {config.node_count}")
    print(f"Relationships: {config.relationship_count}")
    print(f"Text Chunks: {config.chunk_count}")
    print(f"Vector Index: {'Yes' if config.has_vector_index else 'No'}")
    
    if config.sources:
        print(f"\nSource Files:")
        for i, source in enumerate(config.sources, 1):
            print(f"  {i}. {source}")

def update_graph_stats(kg, kg_manager):
    """Update graph statistics in configuration."""
    try:
        # Get current stats
        node_count = kg.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        rel_count = kg.graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
        chunk_count = kg.graph.query("MATCH (c:Chunk) RETURN count(c)").result_set[0][0]
        
        # Check vector index
        has_vector_index = kg.has_vector_index().get("exists", False)
        
        # Load and update config
        config = kg_manager.load_config(kg.name)
        if config:
            config.node_count = node_count
            config.relationship_count = rel_count
            config.chunk_count = chunk_count
            config.has_vector_index = has_vector_index
            kg_manager.save_config(config)
            
    except Exception as e:
        print(f"Warning: Failed to update graph statistics: {e}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced GraphRAG Demo')
    parser.add_argument('--model', type=str, default="gemini/gemini-2.5-flash-preview-05-20",
                        help='LLM to use: gemini/gemini-2.5-flash-preview-05-20, anthropic/claude-3-sonnet-20240229')
    parser.add_argument('--data_dir', type=str, default="data",
                        help='Directory containing data files')
    parser.add_argument('--silent', action='store_true',
                        help='Run in silent mode (suppress most output)')

    parser.add_argument('--ontology_approach', type=str, choices=['flat', 'hierarchical'], default='hierarchical',
                        help='Ontology structure approach: flat (simple) or hierarchical (with inheritance)')
    parser.add_argument('--clean', action='store_true',
                        help='Clear all existing data and start fresh')
    parser.add_argument('--process_documents', '--proc_doc', action='store_true',
                        help='Process documents in the data directory, only if new files are added')
    parser.add_argument('--input_dir', '-i', type=str, default='input',
                        help='Directory containing input files to process.')
    parser.add_argument('--skip_image_analysis', '--skipimg',action='store_true',
                        help='Skip image analysis step. Useful if images are not needed for the current run.')
    parser.add_argument('--output_type', '--out_type', choices=['text', 'json'], default='json',
                        help='Output type for the file preprocessing. Default is JSON. If text is selected, the output will be in raw text file.')
    parser.add_argument('--prompt_for_kg', type=str, default="COT",
                        help='KG extraction prompt type: V1 (default), SG (Schema-Guided), COT (Chain-of-Thought), TD (Enhanced with Evidence)')
    parser.add_argument('--schema_free', action='store_true',
                        help='Disable ontology enforcement during KG extraction (schema-free mode)')
    # NEW REGENERATION OPTIONS
    parser.add_argument('--regenerate-kg', action='store_true',
                        help='Drop and recreate the knowledge graph (preserves in-memory ontology, rebuilds data)')
    parser.add_argument('--regenerate_ontology', action='store_true',
                        help='Force regeneration of ontology from sources even if one exists')
    parser.add_argument('--kg-name', type=str, default=None,
                        help='Custom name for the knowledge graph')
    parser.add_argument('--use-kg', type=str, default=None,
                        help='Use an existing knowledge graph by name')
    parser.add_argument('--ontology-file', type=str, default=None,
                        help='Path to ontology NDJSON to use/save. Defaults to per-KG file.')
    parser.add_argument('--use-update-ontology-prompt', action='store_true',
                        help='Use LLM-based UPDATE_ONTOLOGY_PROMPT to update an existing ontology when new sources are detected')
    parser.add_argument('--list-kgs', action='store_true',
                        help='List all existing knowledge graphs')
    parser.add_argument('--delete-kg', type=str, metavar='KG_NAME',
                        help='Delete a specific knowledge graph')
    parser.add_argument('--kg-info', type=str, metavar='KG_NAME',
                        help='Show information about a specific knowledge graph')
    parser.add_argument('--set-default', type=str, metavar='KG_NAME',
                        help='Set a specific knowledge graph as the default')
    parser.add_argument('--no-ontology-update', action='store_true',
                        help='Do not update ontology from new sources')
    args = parser.parse_args()
    
    # Initialize graph manager
    kg_manager = GraphManager()
    
    # Handle graph management commands
    if args.list_kgs:
        list_knowledge_graphs(kg_manager)
        return
    
    if args.delete_kg:
        delete_knowledge_graph(kg_manager, args.delete_kg)
        return
    
    if args.kg_info:
        show_kg_info(kg_manager, args.kg_info)
        return
    
    if args.set_default:
        success = kg_manager.set_default_graph(args.set_default)
        if success:
            print(f"✓ Set '{args.set_default}' as the default knowledge graph")
        else:
            print(f"✗ Failed to set '{args.set_default}' as default. Graph not found.")
        return
    # Set silent mode if requested
    global SILENT_MODE
    SILENT_MODE = args.silent
    global LANGFUSE_USER
    session_id = f"PROCESS-RAG-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print_only_essential("Starting Enhanced GraphRAG Demo")
    os.environ['SKIP_ANALYZE_IMAGE'] = str(args.skip_image_analysis) # Set environment variable for image analysis
    os.environ['GRAPHRAG_SESSION_ID'] = session_id
    os.environ['LANGFUSE_USER'] = LANGFUSE_USER  # Set Langfuse user ID for tracking
    print_only_essential(f"Starting Enhanced GraphRAG Demo (Session ID: {session_id})")

# Load environment variables
    load_dotenv()
    
    # Configure the model
    model_name = args.model
    print_only_essential(f"Using model: {model_name}")
    
    # First disable all output
    suppress_output()
    
    # Initialize model (potentially verbose)
    try:
        logger.info(f"Model = {model_name}")
        model = initialize_model(model_name)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        print(f"Error initializing model: {e}")
        return
        
    # Determine knowledge graph name with smart default behavior EARLY (needed for ontology path decisions)
    # Initialize graph manager
    kg_manager = GraphManager()

    if args.use_kg:
        kg_name = args.use_kg
    elif args.kg_name:
        kg_name = args.kg_name
    else:
        default_config = kg_manager.get_default_graph()
        kg_name = default_config.name if default_config else "default_graph"

    # Compute ontology file path preference
    ontology_file_path = args.ontology_file or os.path.join("kg_configs", f"{kg_name}_ontology.ndjson")
    os.makedirs(os.path.dirname(ontology_file_path), exist_ok=True)

    # Clear existing data if requested with --clean (removes ALL data including KGs)
    if args.clean:
        print("Clearing ALL existing data to start fresh...")
        try:
            # Connect to Redis to clear all data
            redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT)
            
            # Flush all data
            redis_client.flushall()
            print("Successfully cleared all graph data")
            
            # Remove the ontology file if it exists (global and per-KG)
            removed_any = False
            for path in {"generated_ontology.ndjson", ontology_file_path}:
                if os.path.exists(path):
                    os.remove(path)
                    removed_any = True
            if removed_any:
                print("Removed existing ontology file(s)")
                
        except Exception as e:
            print(f"Warning: Failed to clear existing data: {e}")
    
    # Handle ontology regeneration separately (only removes ontology file, preserves KGs)
    elif args.regenerate_ontology:
        print("Regenerating ontology (preserving existing knowledge graphs)...")
        try:
            # Only remove the ontology file to force regeneration (prefer per-KG path; also remove legacy default if present)
            removed_any = False
            for path in {ontology_file_path, "generated_ontology.ndjson"}:
                if os.path.exists(path):
                    os.remove(path)
                    removed_any = True
            if removed_any:
                print("Removed existing ontology file(s) - will regenerate from sources")
            else:
                print("No existing ontology file found - will generate new one")
        except Exception as e:
            print(f"Warning: Failed to remove ontology file: {e}")
    
    # Allow output briefly to show source discovery
    restore_output()
    
    # Check if the data directory exists
    if args.process_documents:
        # If --process_documents is set, run the data extractor
        print("Processing documents in the data directory...")
        try:
            data_extractor_main(data_dir=Path(args.data_dir), input_path=Path(args.input_dir), output_type=args.output_type )
            print("Document processing completed successfully")
        except Exception as e:
            print(f"Error processing documents: {e}")
            logger.error(f"Filed to process documents: {e}", exc_info=True)
            return
    else:
        print("Skipping document processing as --process_documents is not set. Using existing data.")
        logger.info("Skipping document processing as --process_documents is not set. Using existing data.")

    # Check if using existing knowledge graph - if so, skip data loading
    if args.use_kg:
        print(f"Using existing knowledge graph: {args.use_kg}")
        print("Skipping data loading since using existing knowledge graph...")
        sources = []  # Empty sources list when using existing KG
    else:
        # Define data sources
        data_dir = args.data_dir
        
        # Find and load data sources recursively
        print(f"Scanning for supported files in '{data_dir}' and subdirectories...")
        sources = []
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory '{data_dir}' not found.")
            print(f"Data directory '{data_dir}' not found. Please create it and add documents.")
            return
        
        # Walk through the data directory recursively
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Changed from Source.from_file to Source factory function
                    source = Source(file_path)
                    if source:
                        sources.append(source)
                        print(f"Added source: {file_path}")
                except Exception as e:
                    print(f"Warning: Failed to load source {file_path}: {e}")
        
        if not sources:
            print("No valid sources found. Please add supported documents to the data directory.")
            return
        
        print(f"Successfully loaded {len(sources)} sources")
    
    # Configure model for knowledge graph
    model_config = KnowledgeGraphModelConfig.with_model(model)
    
    # Initialize graph manager (used for both existing and new KGs)
    kg_manager = GraphManager()
    
    # If using existing KG, skip ontology generation but load the existing ontology
    if args.use_kg:
        print("Using existing knowledge graph - skipping ontology generation...")
        # Load existing config
        config = kg_manager.load_config(args.use_kg)
        if not config:
            print(f"Knowledge graph '{args.use_kg}' not found.")
            return
            
        # Load ontology from config
        if os.path.exists(config.ontology_file):
            with open(config.ontology_file, "r", encoding="utf-8") as file:
                lines = file.readlines()
                ontology = Ontology.from_ndjson([json.loads(line) for line in lines])
                print(f"Loaded existing ontology with {len(ontology.entities)} entities and {len(ontology.relations)} relations.")
        else:
            print(f"Ontology file '{config.ontology_file}' not found.")
            return
    else:
        # In schema-free mode, skip ontology generation entirely
        if args.schema_free:
            print("Schema-free mode: skipping ontology generation and using an empty ontology.")
            from graphrag.ontology import Ontology as _Ont
            ontology = _Ont()
        else:
            print("Generating ontology from sources...")
            try:
                # Check if we should regenerate or use existing
                regenerate_ontology = args.regenerate_ontology
                
                # If ontology exists and force regeneration isn't enabled
                if os.path.exists(ontology_file_path) and not regenerate_ontology:
                    # Load existing ontology
                    with open(ontology_file_path, "r", encoding="utf-8") as file:
                        lines = file.readlines()
                        ontology = Ontology.from_ndjson([json.loads(line) for line in lines])
                        logger.info(f"Loaded existing ontology with {len(ontology.entities)} entities and {len(ontology.relations)} relations.")
                        print(f"Loaded existing ontology with {len(ontology.entities)} entities and {len(ontology.relations)} relations.")
                else:
                    # Generate new ontology from sources using LLM and NER
                    print("Generating ontology from sources using LLM and NER...")
                    
                    # Define domain boundaries for healthcare ontology (needed for incremental updates)
                    domain_boundaries = """
                    Extract meaningful entities and relationships from any domain.
                    
                    GUIDANCE FOR ENTITY EXTRACTION:
                    - Identify core entity types that appear across domains: Person, Organization, Document, Location, Event, Process, Concept
                    - Extract domain-specific entity subtypes that inherit from these core types
                    - Capture attributes that describe important properties of each entity
                    
                    GUIDANCE FOR RELATIONSHIP EXTRACTION:
                    - Identify how entities relate to each other with clear, descriptive relationship types
                    - Create hierarchical relationships (IS_A, PART_OF) to establish entity taxonomies
                    - Model cross-domain relationships that connect entities from different domains
                    
                    ONTOLOGY STRUCTURE:
                    - Organize entities in a hierarchical structure with general concepts at the top
                    - Allow specific subtypes to inherit properties from more general types
                    - Balance specificity with generality to capture the essence of the content
                    """
                    
                    try:
                        print("Attempting to generate ontology from sources...")
                        # Use the built-in ontology generation from sources
                        ontology = Ontology.from_sources(
                            sources=sources,
                            model=model,
                            boundaries=domain_boundaries,
                            hide_progress=False,
                            auto_validate=True,
                            language="GERMAN"
                        )
                        
                        print(f"Successfully generated dynamic ontology with {len(ontology.entities)} entities and {len(ontology.relations)} relations.")
                        
                        # Save the generated ontology in NDJSON format
                        with open(ontology_file_path, "w", encoding="utf-8") as file:
                            for entity in ontology.entities:
                                file.write(json.dumps({"type": "entity_definition", **entity.to_json()}) + "\n")
                            for relation in ontology.relations:
                                file.write(json.dumps({"type": "relation_definition", **relation.to_json()}) + "\n")
                        
                        print("Saved generated ontology to file for future use.")
                        
                        # Print the generated entities and relations for debugging
                        print("\nGenerated Entity Types:")
                        for entity in ontology.entities:
                            print(f"  - {entity.label} with attributes: {', '.join([attr.name for attr in entity.attributes])}")
                        
                        print("\nGenerated Relationship Types:")
                        for relation in ontology.relations:
                            print(f"  - {relation.label}: {relation.source.label} -> {relation.target.label}")
                            
                    except Exception as e:
                        # Print the full error details
                        print(f"\nERROR during ontology generation: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        
                        # Log detailed error information for debugging
                        logger.error(f"Ontology generation failed completely. Error type: {type(e).__name__}")
                        logger.error(f"Error message: {str(e)}")
                        logger.error(f"Error details: {traceback.format_exc()}")
                        logger.error("No fallback ontology will be created. System will exit.")
                        
                        print(f"\n❌ Ontology generation failed: {type(e).__name__}: {e}")
                        print("   No fallback ontology will be created.")
                        print("   Please check your data sources and model configuration.")
                        print("   System will exit.")
                        
                        # Exit the system since ontology is critical
                        sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to generate ontology: {e}", exc_info=True)
                print(f"Error generating ontology: {e}")
                sys.exit(1)
    
    # Display ontology details
    print("\n=== ONTOLOGY STRUCTURE ===")
    for entity in ontology.entities:
        print(f"Entity: {entity.label} - Attributes: {len(entity.attributes)}")
    
    for relation in ontology.relations:
        print(f"Relation: {relation.label} - {relation.source.label} -> {relation.target.label}")
    print("=========================\n")
    
    restore_output()
    print(f"Ontology ready. Using file: {ontology_file_path}")
    print("Creating knowledge graph...")
    suppress_output()
    
    # At this point, kg_name is already determined above
    if args.use_kg:
        print(f"Using existing knowledge graph: {kg_name}")
    elif args.kg_name:
        print(f"Creating new knowledge graph: {kg_name}")
    else:
        print(f"Using default knowledge graph: {kg_name}")

    # Create knowledge graph
    config_file_path = f"kg_configs/{kg_name}_config.json"
    kg = KnowledgeGraph(
        name=kg_name,
        model_config=model_config,
        ontology=ontology,
        host=REDIS_HOST,
        port=REDIS_PORT,
        enable_embeddings=True,
        qa_prompt=MULTI_LEVEL_QA_PROMPT,
        qa_system_instruction=MULTI_LEVEL_QA_SYSTEM,
        config_file_path=config_file_path
    )
    

    kg.refresh_schema()
    print("Schema refreshed successfully!")
    # Set verbose logging for debugging
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.INFO)
        logging.getLogger(name).propagate = True
    
    # Process sources to populate the graph
    start_time = time.time()
    print("Processing sources to populate knowledge graph...")
    
    # Define domain boundaries for healthcare ontology (needed for incremental updates)
    domain_boundaries = """
    Extract meaningful entities and relationships from any domain.
    
    GUIDANCE FOR ENTITY EXTRACTION:
    - Identify core entity types that appear across domains: Person, Organization, Document, Location, Event, Process, Concept
    - Extract domain-specific entity subtypes that inherit from these core types
    - Capture attributes that describe important properties of each entity
    
    GUIDANCE FOR RELATIONSHIP EXTRACTION:
    - Identify how entities relate to each other with clear, descriptive relationship types
    - Create hierarchical relationships (IS_A, PART_OF) to establish entity taxonomies
    - Model cross-domain relationships that connect entities from different domains
    
    ONTOLOGY STRUCTURE:
    - Organize entities in a hierarchical structure with general concepts at the top
    - Allow specific subtypes to inherit properties from more general types
    - Balance specificity with generality to capture the essence of the content
    """
    
    # ENHANCED SOURCE PROCESSING LOGIC
    # Skip source processing when using existing KG
    if args.use_kg:
        print("Using existing knowledge graph - skipping source processing...")
        new_sources = []  # No new sources to process
    elif args.regenerate_kg:
        print("🔄 Regenerating knowledge graph (drop + recreate)...")
        # Drop and recreate graph via helper; removes stale labels and data, preserves in-memory ontology
        clear_kg_data_preserve_ontology(kg)
        # Force reprocess all sources
        kg.processed_sources = set()
        new_sources = sources
        print(f"✅ Will reprocess all {len(sources)} sources")
        
        
    else:
        # Original incremental processing logic
        kg._load_processed_sources()
        already_processed = set(kg.processed_sources)
        
        # Identify new sources
        new_sources = []
        for source in sources:
            source_path = getattr(source, 'data_source', str(source))
            if source_path not in already_processed:
                new_sources.append(source)
        
        if new_sources:
            print(f"Found {len(new_sources)} new sources to process...")
        else:
            print("No new sources to process - knowledge graph is up to date")
            print("💡 Use --regenerate-kg to regenerate with preserved ontology")

    # Process sources if we have any
    if new_sources:
        # If there are new sources, check if we need to update the ontology
        if len(new_sources) > 0 and not args.schema_free and not args.regenerate_ontology and not args.regenerate_kg and not args.no_ontology_update:
            print("Updating ontology with new sources...")
            try:
                if args.use_update_ontology_prompt:
                    # Prepare existing ontology NDJSON string
                    existing_lines = []
                    for e in ontology.entities:
                        existing_lines.append(json.dumps({"type": "entity_definition", **e.to_json()}))
                    for r in ontology.relations:
                        existing_lines.append(json.dumps({"type": "relation_definition", **r.to_json()}))
                    existing_ndjson = "\n".join(existing_lines)

                    # Concatenate new sources' text
                    combined_texts = []
                    for s in new_sources:
                        try:
                            docs = list(s.load())
                            for d in docs:
                                if hasattr(d, 'content') and d.content:
                                    combined_texts.append(d.content)
                        except Exception:
                            continue
                    new_text = "\n\n".join(combined_texts)

                    updated = update_ontology_via_llm(model, existing_ndjson, new_text, boundaries=domain_boundaries, language="GERMAN")
                    if updated:
                        ontology = ontology.merge_with(updated)
                        print("Applied LLM-based ontology update via UPDATE_ONTOLOGY_PROMPT")
                    else:
                        print("LLM-based ontology update failed; falling back to schema-guided merge")
                        temp_ontology = Ontology.from_sources(
                            sources=new_sources,
                            model=model,
                            boundaries=domain_boundaries,
                            hide_progress=False,
                            auto_validate=True,
                            language="GERMAN"
                        )
                        ontology.merge_with(temp_ontology)
                else:
                    # Fallback: generate temporary ontology and merge
                    temp_ontology = Ontology.from_sources(
                        sources=new_sources,
                        model=model,
                        boundaries=domain_boundaries,
                        hide_progress=False,
                        auto_validate=True,
                        language="GERMAN"
                    )
                    ontology.merge_with(temp_ontology)

                # Save updated ontology
                with open(ontology_file_path, "w", encoding="utf-8") as file:
                    for entity in ontology.entities:
                        file.write(json.dumps({"type": "entity_definition", **entity.to_json()}) + "\n")
                    for relation in ontology.relations:
                        file.write(json.dumps({"type": "relation_definition", **relation.to_json()}) + "\n")
            except Exception as e:
                print(f"Warning: Failed to update ontology with new sources: {e}")
                print("Continuing with existing ontology...")
        
        # Process new sources
        try:
            print(f"⚙️ Processing {len(new_sources)} sources...")
            kg.process_sources(new_sources, prompt_type=args.prompt_for_kg, language="GERMAN", use_ontology=not args.schema_free)
            print(f"✓ Successfully processed {len(new_sources)} sources")
        except Exception as e:
            print(f"✗ Error processing sources: {str(e)}")
            print(f"  Error details: {type(e).__name__}")
            traceback.print_exc()
            
    end_time = time.time()
    
    # Print only essential info
    restore_output()
    print(f"Graph processing completed in {end_time - start_time:.2f} seconds")
    #print(f"Ontology has {len(ontology.entities)} entities and {len(ontology.relations)} relations.")
    
    # Get graph stats - but suppress output of the query itself
    suppress_output()
    graph = kg.graph
    stats = graph.query("MATCH (n) RETURN count(n) as nodes").result_set[0][0]
    rel_stats = graph.query("MATCH ()-[r]->() RETURN count(r) as relationships").result_set[0][0]
    
    # Restore chunk stats
    chunk_stats = graph.query("MATCH (c:Chunk) RETURN count(c) as chunks").result_set[0][0]
    
    restore_output()
    print("\n=== KNOWLEDGE GRAPH STATISTICS ===")
    print(f"Total nodes: {stats}")
    print(f"Total relationships: {rel_stats}")
    # Restore embedding-related information
    print(f"Total text chunks for embedding: {chunk_stats}")
    
    # Restore vector index check
    suppress_output()
    index_query = "CALL db.indexes() YIELD label, properties, options WHERE label = 'Chunk' AND 'embedding' IN properties RETURN options"
    index_result = graph.query(index_query)
    
    restore_output()
    if index_result and index_result.result_set and len(index_result.result_set) > 0:
        print("Vector index created for fast similarity search")
    
    restore_output()
    print_only_essential("Knowledge graph populated successfully")
    
    # Save graph configuration after processing
    if not args.use_kg:
        # Determine if this should be the default graph
        is_default = (kg_name == "default_graph" or 
                     (not args.kg_name and kg_manager.get_default_graph() is None))
        
        config = GraphConfig(
            name=kg_name,
            model_name=model_name,
            ontology_file=ontology_file_path,
            sources=[getattr(s, 'data_source', str(s)) for s in sources],
            is_default=is_default
        )
        kg_manager.save_config(config)
        print(f"Saved configuration for knowledge graph: {kg_name}")
        print(f"Using ontology file: {ontology_file_path}")
        if is_default:
            print(f"Set as default graph: {kg_name}")
    
    # Update graph statistics
    update_graph_stats(kg, kg_manager)

    print_only_essential("\nStarting interactive chat session. Type 'exit' to quit.")
    # Start chat session (non-embedding based)
    chat_session = kg.chat_session()

    while True:
        question = input("\nYour question (or 'exit' to quit): ").strip()

        if question.lower() == 'exit':
            break
        if not question:
            continue

        print_only_essential("\nProcessing your question...")
        
        response = None # Initialize response
        query_time = 0
        
        try:
            # Set a timeout for the request
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # 2 minutes timeout

            start_time_processing = time.time()

            # Use the knowledge graph's chat method directly
            response = kg.chat(question)
            
            # Check if GraphRAG returned its own structured error
            # This is a heuristic based on observed error messages.
            is_graphrag_internal_error = False
            if isinstance(response, dict):
                brief_answer = response.get('brief_answer', '')
                detailed_info = response.get('detailed_info', '')
                if brief_answer == "Error processing request" and \
                   ("error occurred" in detailed_info.lower() or "'query'" in detailed_info):
                    is_graphrag_internal_error = True

            if is_graphrag_internal_error:
                print_only_essential("GraphRAG component returned an internal error.")
                response = {
                    "brief_answer": "Error processing request",
                    "detailed_info": "The GraphRAG system encountered an internal error while processing your question.",
                    "sources": [],
                    
                }

            query_time = time.time() - start_time_processing
            signal.alarm(0)  # Disable the alarm
            
            print_only_essential(f"Query processed in {query_time:.2f} seconds")

            # Display the structured response
            if response: # Ensure response is not None
                 display_structured_response(response)
            else:
                # This case should ideally not be reached if kg.chat() always returns something
                print_only_essential("No response was generated.")

        except TimeoutError:
            signal.alarm(0)  # Disable the alarm
            print_only_essential("\nError: The request timed out.")
            # Ensure response is a dict for display_structured_response
            response = {"brief_answer": "Timeout", "detailed_info": "Request timed out.", "sources": [], "requires_verification": True, "verification_reason": "Timeout", "has_gaps": True, "alternatives": []}
            display_structured_response(response)
        except Exception as e:
            signal.alarm(0)  # Disable the alarm
            print_only_essential(f"\nERROR in processing: {type(e).__name__}: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            # Ensure response is a dict for display_structured_response
            response = {"brief_answer": "Critical Error", "detailed_info": f"Unhandled error: {str(e)}", "sources": [], "requires_verification": True, "verification_reason": "System error", "has_gaps": True, "alternatives": []}
            display_structured_response(response)

    print_only_essential("\nExiting chat session.")

def initialize_and_process_query(question, kg_name, parsed_model_name=None):
    """
    Initialize the system and process a query using the knowledge graph.
    """
    try:
        # Load environment variables
        load_dotenv()

        # Configure the model - use parsed_model_name if provided, otherwise use default
        model_name = parsed_model_name if parsed_model_name else "gemini/gemini-2.5-flash-preview-05-20"
        model = initialize_model(model_name)
        print(f"MODEL NAME ==> {model_name}")
        
        # Load or generate the ontology
        

        # Initialize the knowledge graph
        model_config = KnowledgeGraphModelConfig.with_model(model)
        config_file_path = "kg_configs/knowledge_graph_config.json"
        kg = KnowledgeGraph(
            name="WUP_FINAL2",
            model_config=model_config,
            host=REDIS_HOST,
            port=REDIS_PORT,
            enable_embeddings=True,
            qa_prompt=MULTI_LEVEL_QA_PROMPT,
            qa_system_instruction=MULTI_LEVEL_QA_SYSTEM,
            config_file_path=config_file_path
        )

        # Process the query
        response = kg.chat(question)

        

        return response

    except Exception as e:
        print(f"Error in initialize_and_process_query: {e}")
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    main() 