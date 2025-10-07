from .pdf import PDFLoader
from .text import TextLoader
from .html import HTMLLoader
from .url import URLLoader
from .jsonl import JSONLLoader
from .json_loader import JSONLoader
from .docx_loader import DOCXLoader

__all__ = [
    "PDFLoader",
    "TextLoader",
    "HTMLLoader",
    "URLLoader",
    "JSONLLoader",
    "JSONLoader",
    "DOCXLoader"
]
