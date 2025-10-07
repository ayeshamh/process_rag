from typing import Iterator, Optional
from .document import Document
from .base import AbstractSource
# Import standard loaders from local package
from .document_loaders.pdf import PDFLoader
from .document_loaders.text import TextLoader
from .document_loaders.url import URLLoader
from .document_loaders.html import HTMLLoader
from .document_loaders.jsonl import JSONLLoader
# Import custom loaders from local package
from .document_loaders.json_loader import JSONLoader
from .document_loaders.docx_loader import DOCXLoader

SUPPORTED_EXTENSIONS = {
    ".pdf": PDFLoader,
    ".txt": TextLoader,
    ".html": HTMLLoader,
    ".json": JSONLoader,
    ".jsonl": JSONLLoader,
    ".docx": DOCXLoader,
}

def Source(path: str) -> "AbstractSource":
    """
    Creates a source object

    Args:
        path (str): path to source
        
    Returns:
        AbstractSource: A source object corresponding to the input path format.
    """
    if not isinstance(path, str) or path == "":
        raise Exception("Invalid argument, path should be a none empty string.")

    if ".pdf" in path.lower():
        s = PDF(path)
    elif ".html" in path.lower():
        s = HTML(path)
    elif "http" in path.lower():
        s = URL(path)
    elif ".jsonl" in path.lower():
        s = JSONL(path)
    elif ".txt" in path.lower():
        s = TEXT(path)
    elif ".json" in path.lower():
        s = JSON(path)
    elif ".docx" in path.lower() or ".doc" in path.lower():
        s = DOCX(path)
    else:
        raise Exception("Unsupported file format.")

    return s

class PDF(AbstractSource):
    """
    PDF resource
    """
    def __init__(self, data_source):
        super().__init__(data_source)
        self.loader = PDFLoader(self.data_source)

class TEXT(AbstractSource):
    """
    TEXT resource
    """
    def __init__(self, data_source):
        super().__init__(data_source)
        self.loader = TextLoader(self.data_source)

class URL(AbstractSource):
    """
    URL resource
    """
    def __init__(self, data_source):
        super().__init__(data_source)
        self.loader = URLLoader(self.data_source)

class HTML(AbstractSource):
    """
    HTML resource
    """
    def __init__(self, data_source):
        super().__init__(data_source)
        self.loader = HTMLLoader(self.data_source)

class JSONL(AbstractSource):
    """
    JSONL resource
    """
    def __init__(self, data_source, rows_per_document: int = 50):
        super().__init__(data_source)
        self.loader = JSONLLoader(self.data_source, rows_per_document)

class JSON(AbstractSource):
    """
    JSON/NDJSON resource loader
    Handles both regular JSON and NDJSON formats
    """
    def __init__(self, data_source: str):
        """
        Initialize JSON source loader
        
        Args:
            data_source (str): Path to the JSON/NDJSON file
        """
        super().__init__(data_source)
        self.loader = JSONLoader(self.data_source)
        
    def load(self) -> Iterator[Document]:
        """
        Load the JSON/NDJSON file and yield Document objects.
        Automatically handles conversion between formats.
        
        Returns:
            Iterator[Document]: Iterator of Document objects
        """
        content = self.loader.load()
        if content:
            yield Document(
                id=self.data_source,
                content=content
            )

class DOCX(AbstractSource):
    """
    DOCX resource
    """
    def __init__(self, data_source):
        super().__init__(data_source)
        self.loader = DOCXLoader(self.data_source)

class Source_FromRawText(AbstractSource):
    """
    Create a source from raw text content
    """
    def __init__(self, text_content: str, uri: Optional[str] = None):
        """
        Initialize a source directly from raw text.
        
        Args:
            text_content (str): The raw text content
            uri (Optional[str]): Optional identifier for the source
        """
        super().__init__(uri or "raw_text_content")
        self.content = text_content
        
    def load(self) -> Iterator[Document]:
        """
        Return the raw text content as a Document.
        
        Returns:
            Iterator[Document]: Iterator with a single Document containing the text
        """
        doc = Document(self.content, id=self.data_source)
        yield doc
