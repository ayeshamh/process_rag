import logging
from typing import List, Dict, Any
from .Step import Step
from ..models import GenerativeModelChatSession


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class QAStep(Step):
    """
    QA step
    """
    
    def __init__(self, chat_session: GenerativeModelChatSession, qa_prompt: str) -> None:
        """
        Initialize step

        Args:
            chat_session (GenerativeModelChatSession): Chat session to use.
            qa_prompt (str): Prompt for QA. Required parameter.
        """
        self.chat_session = chat_session
        self.qa_prompt = qa_prompt

    def run(self, question: str, context: List[Dict[str, Any]],
            chunk_context: List[Dict[str, Any]] | None = None,
            entity_context: List[Dict[str, Any]] | None = None,
            graph_neighbors: List[Dict[str, Any]] | None = None) -> str:
        """
        Run step

        Args:
            question (str): Question to ask.
            context (List[Dict[str, Any]]): Context for question.

        Returns:
            str: Answer to question.
        """
        # Sanitize to exclude embedding-like fields from any dict/list inputs
        def _sanitize(obj: Any):
            try:
                # keys to drop at any nesting level
                drop_keys = {"node_embedding", "embedding", "_embedding", "vector", "embedding_vector"}
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items() if k not in drop_keys}
                if isinstance(obj, list):
                    return [_sanitize(v) for v in obj]
                return obj
            except Exception:
                return obj

        clean_context = _sanitize(context or [])
        clean_chunk_context = _sanitize(chunk_context or [])
        clean_entity_context = _sanitize(entity_context or [])
        clean_graph_neighbors = _sanitize(graph_neighbors or [])

        # Format context data into readable text for the model
        def _format_context_as_text(data, context_type=""):
            if not data:
                return f"[No {context_type} available]"
          
            if isinstance(data, list):
                if not data:
                   return f"[No {context_type} available]"
              
                formatted_items = []
                for i, item in enumerate(data, 1):
                    if isinstance(item, dict):
                        if context_type == "chunk_context":
                           # Format chunk context
                           text = item.get('text', '')
                           source = item.get('source', 'Unknown source')
                           score = item.get('score', 0)
                           formatted_items.append(f"[{i}] Source: {source} | \nText: {text} | Score: {score}")
                        elif context_type == "entity_context":
                           # Format entity context
                           name = item.get('name', '')
                           labels = '/'.join(item.get('labels', []))
                           description = item.get('description', '')
                           score = item.get('score', 0)
                           formatted_items.append(f"[{i}] {labels}: {name} | \nDescription: {description}")
                        elif context_type == "graph_neighbors":
                           # Format graph neighbors with names instead of IDs
                           src_name = item.get('src_name', '')
                           src_id = item.get('src', '')
                           src_desc = item.get('src_description', '')
                           rel = item.get('rel', '')
                           dst_name = item.get('dst_name', '')
                           dst_id = item.get('dst', '')
                           dst_desc = item.get('dst_description', '')
                           rel_props = item.get('rel_props', {})
                           source_doc = rel_props.get('source_document', '')

                           # Use name if available, otherwise fall back to ID
                           src_display = src_name if src_name else f"ID:{src_id}"
                           dst_display = dst_name if dst_name else f"ID:{dst_id}"
                           
                           # Build base string
                           base_string = f"[{i}] {src_display}: {src_desc} ----[{rel}]----> {dst_display} : {dst_desc}"
                           
                           # Add source document if available
                           if source_doc:
                               base_string += f" | Source: {source_doc}"
                           
                           formatted_items.append(base_string)
                        else:
                           # Generic formatting
                           formatted_items.append(f"[{i}] {str(item)}")
                    else:
                       formatted_items.append(f"[{i}] {str(item)}")
              
                return '\n\n'.join(formatted_items)

            else:
                return str(data)


       # Format each context type as readable text
        formatted_context = _format_context_as_text(clean_context, "context")
        formatted_chunk_context = _format_context_as_text(clean_chunk_context, "chunk_context")
        formatted_entity_context = _format_context_as_text(clean_entity_context, "entity_context")
        formatted_graph_neighbors = _format_context_as_text(clean_graph_neighbors, "graph_neighbors")


        formatted_prompt = self.qa_prompt.format(
           context=formatted_context,
           question=question,
           chunk_context=formatted_chunk_context,
           entity_context=formatted_entity_context,
           graph_neighbors=formatted_graph_neighbors
        )


        logger.info(f"🔍 QA Step Formatted Prompt: {formatted_prompt}")
        # Ensure the formatted prompt is not empty

        if not formatted_prompt or not formatted_prompt.strip():
            logger.info(f"🔴 QA step received empty prompt")
            formatted_prompt = f"Please answer this question: {question}"
        
        response = self.chat_session.send_message(formatted_prompt)
        
        # Log response details for debugging
        if hasattr(response, 'text'):
            if not response.text or response.text.strip() == "":
                logger.warning(f"🔴 QA step received empty response from chat session")
            elif response.text.strip() == "No response generated":
                logger.warning(f"🔴 QA step received 'No response generated' from chat session")
            else:
                logger.debug(f"✅ QA step received response: {response.text[:100]}...")
        
        return response
