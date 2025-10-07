from abc import ABC
from typing import Iterator, Optional
from .document import Document

class AbstractSource(ABC):
    """
    Abstract class representing a source file
    """

    def __init__(self, data_source: str):
        """
        Initializes a new instance of the Source class.

        Args:
            data_source (str): Either a file path or a string.

        Attributes:
            data_source (str): The source path for the data or the data as a string.
            loader: The loader object associated with the source.
            source_type (str): The type of the source.
        """
        self.data_source = data_source
        self.loader = None
        self.source_type = self.__class__.__name__

    def load(self) -> Iterator[Document]:
        """
        Loads documents from the source.

        Returns:
            An iterator of Document objects.
        """
        return self.loader.load()

    def __eq__(self, other) -> bool:
        """
        Check if this source object is equal to another source object.

        Args:
            other: The other source object to compare with.

        Returns:
            bool: True if the source objects are equal, False otherwise.
        """
        if not isinstance(other, AbstractSource):
            return False

        return self.data_source == other.data_source

    def __hash__(self):
        """
        Calculates the hash value of the Source object based on its data_source.

        Returns:
            int: The hash value of the Source object.
        """
        return hash(self.data_source) 