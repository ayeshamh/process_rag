# GraphRAG Examples

This directory contains examples to help you get started with the GraphRAG framework.

## General RAG Demo

The `general_rag_demo.py` script demonstrates a complete GraphRAG workflow:

1. Loading documents from a directory
2. Automatically extracting ontology
3. Creating a knowledge graph
4. Generating embeddings for semantic search
5. Starting an interactive chat session

### Running the Demo

```bash
# Start the FalkorDB server (in a separate terminal)
docker run -p 6379:6379 -p 3000:3000 -it --rm -v /path/to/your/data:/data falkordb/falkordb:latest

# Create a .env file with your API keys
# Example:
# GOOGLEAI_API_KEY=your_api_key
# ANTHROPIC_API_KEY=your_api_key

# Install required dependencies
pip install requirements.txt

# Run the demo
python examples/general_rag_demo.py
```


### Running with Input Files

To run the file and process input files, use the following command:

```bash
python examples/general_rag_demo.py -i "Input" --process_documents
```

Here:

-   `-i` denotes the input folder location relative to the base directory.
-   `--process_documents` is a flag indicating that the documents should be processed.

By default, the output is saved in the "data" folder.


### Features

- **Automatic Document Processing**: Handles various file formats (JSON, DOCX, PDF, etc.)
- **Ontology Generation**: Creates an ontology structure from your documents
- **Knowledge Graph Creation**: Transforms unstructured data into a graph database
- **Multi-Level Responses**: Provides concise answers, detailed information, and source references
- **Verification Detection**: Flags queries that require human verification
- **Gap Handling**: Identifies missing information and suggests alternatives
- **Semantic Search**: Find relevant document chunks using embedding similarity

### Customization Options

You can customize various aspects of the demo:

- Change the LLM model by modifying the `model_name` variable:
  ```python
  # Use Gemini
  model_name = "gemini/gemini-2.0-flash"
  
  # Use Claude
  model_name = "anthropic/claude-3-opus"
  ```
- Set data directory with an environment variable `DATA_DIR`
- Configure embedding settings (model, chunk size, overlap)


### Example Queries

Try asking questions like:

- "What is the process for enrolling a new resident?"
- "Is there availability for two people with dementia in the nursing home?"
- "What documents are required for admission?"
- "semantic: Tell me about dementia care" (uses semantic search)

## Common Issues

- **No documents found**: Ensure your data directory contains supported file formats
- **FalkorDB connection error**: Check that the FalkorDB server is running
- **Model API errors**: Verify your API keys are correctly set in environment variables
- **Unsupported file format errors**: Check that your file extensions match one of the supported formats
- **Issues with DOCX files**: Make sure you have the python-docx package installed
- **Issues with embeddings**: Ensure sentence-transformers is properly installed

## Different Search Types


##  Ontology Creation with spaCy

The system now uses spaCy's Named Entity Recognition (NER) to improve ontology creation. This enhancement:

1. Pre-processes documents with spaCy to identify potential entities
2. Uses these entities to guide the LLM in creating a more comprehensive ontology
3. Improves coverage of domain-specific entities that might be missed otherwise

### Setup for  Ontology Creation

1. Install spaCy and download a model:

```bash
# First ensure spaCy is installed (added to requirements.txt)
pip install -r requirements.txt

# Download a spaCy model (larger models give better results but require more memory)
python -m spacy download en_core_web_lg  # Recommended for best results
# OR for faster but less accurate NER: 
# python -m spacy download de_core_news_sm
```

2. Configure the model in your code (optional - defaults to en_core_web_lg):

```python
# When creating your ontology, you can specify which spaCy model to use:
step = CreateOntologyStep(
    sources=sources,
    ontology=Ontology(),
    model=model,
    config={"spacy_model": "en_core_web_lg"},  # Or "de_core_news_sm" for faster processing
    hide_progress=hide_progress,
    language="en",  # Language code for extraction (e.g., "en", "de")
)

