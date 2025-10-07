import json
import logging
import os
from typing import Dict, Any, List, Optional, Union
from ..helpers import parse_ndjson

logger = logging.getLogger(__name__)

class JSONLoader:
    """
    Loads JSON files and extracts their content.
    Enhanced to handle nested JSON structures and convert to NDJSON format.
    """

    def __init__(self, file_path: str):
        """
        Initialize the JSON loader.
        
        Args:
            file_path (str): Path to the JSON file
        """
        self.file_path = file_path
        self.extracted_text = ""
        self.structured_data = None
        
    def _flatten_json(self, data: Union[Dict, List], parent_key: str = '') -> List[Dict]:
        """
        Flattens nested JSON structures into a list of NDJSON-compatible dictionaries.
        
        Args:
            data: The JSON data to flatten
            parent_key: The parent key for nested structures
            
        Returns:
            List of flattened dictionaries
        """
        flattened = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                
                if isinstance(value, (dict, list)):
                    flattened.extend(self._flatten_json(value, new_key))
                else:
                    flattened.append({"type": "entity", "key": new_key, "value": value})
                    
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{parent_key}[{i}]" if parent_key else str(i)
                flattened.extend(self._flatten_json(item, new_key))
                
        return flattened
        
    def load(self) -> str:
        """
        Loads and returns the content of a regular JSON file as a string.
        Only supports standard JSON, not NDJSON.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            # Validate it's proper JSON
            json.loads(content)
            return content
        except Exception as e:
            logger.error(f"Error loading JSON file {self.file_path}: {str(e)}")
            return "" 