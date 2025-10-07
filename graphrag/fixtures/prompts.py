"""Compatibility stub for prompts. Use prompts_for_KG or prompts_for_RAG instead."""
from .prompts_for_KG import *  # noqa
from .prompts_for_RAG import *  # noqa

# Backward compatibility aliases for the default prompts
from .prompts_for_KG import EXTRACT_DATA_SYSTEM_COT as EXTRACT_DATA_SYSTEM
from .prompts_for_KG import EXTRACT_DATA_PROMPT_COT as EXTRACT_DATA_PROMPT