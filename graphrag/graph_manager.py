from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import os

class GraphConfig(BaseModel):
    """Configuration for a knowledge graph."""
    name: str = Field(..., description="Unique name for the knowledge graph")
    created_at: datetime = Field(default_factory=datetime.now)
    model_name: str = Field(..., description="LLM model used")
    ontology_file: str = Field(..., description="Path to ontology file")
    sources: List[str] = Field(default_factory=list, description="Source files used")
    node_count: int = Field(default=0, description="Number of nodes")
    relationship_count: int = Field(default=0, description="Number of relationships")
    chunk_count: int = Field(default=0, description="Number of text chunks")
    has_vector_index: bool = Field(default=False, description="Whether vector index exists")
    language: str = Field(default="de", description="Language code for processing (e.g., 'en', 'de')")
    is_default: bool = Field(default=False, description="Whether this is the default graph")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class GraphManager:
    """Simple, efficient graph management."""
    
    def __init__(self, config_dir: str = "kg_configs"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, config: GraphConfig) -> None:
        """Save graph configuration."""
        config_file = os.path.join(self.config_dir, f"{config.name}_config.json")
        with open(config_file, "w") as f:
            json.dump(config.dict(), f, indent=2, default=str)
    
    def load_config(self, name: str) -> Optional[GraphConfig]:
        """Load graph configuration."""
        config_file = os.path.join(self.config_dir, f"{name}_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                    return GraphConfig(**data)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Invalid config file {config_file}: {e}")
                return None
        return None
    
    def list_graphs(self) -> List[GraphConfig]:
        """List all available graphs."""
        configs = []
        for file in os.listdir(self.config_dir):
            if file.endswith("_config.json"):
                name = file.replace("_config.json", "")
                config = self.load_config(name)
                if config:
                    configs.append(config)
        return sorted(configs, key=lambda x: x.created_at, reverse=True)
    
    def delete_config(self, name: str) -> bool:
        """Delete graph configuration."""
        config_file = os.path.join(self.config_dir, f"{name}_config.json")
        if os.path.exists(config_file):
            os.remove(config_file)
            return True
        return False
    
    def get_default_graph(self) -> Optional[GraphConfig]:
        """Get the default graph configuration."""
        configs = self.list_graphs()
        for config in configs:
            if config.is_default:
                return config
        return None
    
    
    def set_default_graph(self, name: str) -> bool:
        """Set a specific graph as default (unsets others)."""
        # Unset all other defaults
        configs = self.list_graphs()
        for config in configs:
            if config.is_default and config.name != name:
                config.is_default = False
                self.save_config(config)
        
        # Set the specified graph as default
        target_config = self.load_config(name)
        if target_config:
            target_config.is_default = True
            self.save_config(target_config)
            return True
        return False