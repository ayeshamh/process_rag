"""
KG extraction prompt selector - clean, type-safe implementation following Pydantic patterns.
"""
from typing import Tuple, List
from enum import Enum
from dataclasses import dataclass
from . import prompts_for_KG as prompts_module


class PromptType(str, Enum):
    """Available KG extraction prompt types."""
    V1 = "V1"      # Version 1 (Original)
    SG = "SG"      # Schema-Guided
    COT = "COT"    # Chain-of-Thought
    TD = "TD"    # Enhanced Version 1 with Evidence Tracking
    
    @classmethod
    def list_available(cls) -> List[str]:
        """Return list of available prompt types."""
        return [item.value for item in cls]


@dataclass(frozen=True)
class PromptPair:
    """Immutable container for system and user prompts."""
    system: str
    user: str
    
    def __post_init__(self):
        """Validate prompts are non-empty."""
        if not self.system or not self.user:
            raise ValueError("Both system and user prompts must be non-empty")


class PromptSelector:
    """Type-safe prompt selector with validation and error handling."""
    
    _SYSTEM_PREFIX = "EXTRACT_DATA_SYSTEM_"
    _USER_PREFIX = "EXTRACT_DATA_PROMPT_"
    
    @classmethod
    def get_prompt_pair(cls, prompt_type: str) -> PromptPair:
        """
        Get validated prompt pair for the specified type.
        
        Args:
            prompt_type: The prompt type (e.g., 'V1', 'SG', 'COT')
            
        Returns:
            PromptPair: Validated system and user prompts
            
        Raises:
            ValueError: If prompt type is invalid or prompts are missing
        """
        # Normalize input
        normalized_type = prompt_type.upper().strip()
        
        # Validate type exists in enum
        try:
            PromptType(normalized_type)
        except ValueError:
            available = PromptType.list_available()
            raise ValueError(
                f"Invalid prompt type '{prompt_type}'. "
                f"Available options: {', '.join(available)}"
            ) from None
        
        # Build constant names
        system_name = f"{cls._SYSTEM_PREFIX}{normalized_type}"
        user_name = f"{cls._USER_PREFIX}{normalized_type}"
        
        # Get prompts with proper error handling
        try:
            system_prompt = getattr(prompts_module, system_name)
            user_prompt = getattr(prompts_module, user_name)
        except AttributeError as e:
            missing_attr = str(e).split("'")[1]
            raise ValueError(
                f"Prompt constant '{missing_attr}' not found in prompts_for_KG module. "
                f"Please ensure both {system_name} and {user_name} are defined."
            ) from None
        
        # Return validated pair
        return PromptPair(system=system_prompt, user=user_prompt)
    
    @classmethod
    def list_available_prompts(cls) -> List[str]:
        """Get list of all available prompt types."""
        return PromptType.list_available()


# Convenience function for backward compatibility
def get_prompt_pair(prompt_type: str) -> Tuple[str, str]:
    """
    Convenience function returning tuple for compatibility.
    
    Args:
        prompt_type: The prompt type (e.g., 'V1', 'SG', 'COT')
        
    Returns:
        Tuple[str, str]: (system_prompt, user_prompt)
    """
    pair = PromptSelector.get_prompt_pair(prompt_type)
    return pair.system, pair.user