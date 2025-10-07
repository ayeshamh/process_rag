import logging
import os
from typing import Optional, Iterator, Dict, Any, List, Union
from litellm import completion, validate_environment,completion_cost, utils as litellm_utils


from .model import (
    OutputMethod,
    GenerativeModel,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
    GenerativeModelChatSession,
)

from langfuse import observe, get_client
from datetime import datetime
langfuse = get_client()  # Initialize Langfuse client for tracing
session_id =  f"default-session-{datetime.now().strftime('%Y%m%d')}"
langfuse_user_id = "default-user"  # Default user ID for Langfuse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

class LiteModel(GenerativeModel):
    """
    A generative model that interfaces with the LiteLLM for chat completions.
    """

    def __init__(
        self,
        model_name: str,
        generation_config: Optional[GenerativeModelConfig] = None,
        system_instruction: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LiteModel with the required parameters.
        
        LiteLLM model_name format: <provider>/<model_name>
         Examples:
         - openai/gpt-4o
         - gemini/gemini-1.5-pro
         - anthropic/claude-3-opus

        Args:
            model_name (str): The name and the provider for the LiteLLM client.
            generation_config (Optional[GenerativeModelConfig]): Configuration settings for generation.
            system_instruction (Optional[str]): Instruction to guide the model.
            api_key (Optional[str]): Optional API key for the model provider.
            **kwargs: Additional keyword arguments to pass to LiteLLM.
        """
        global session_id, langfuse_user_id
        session_id = os.getenv('GRAPHRAG_SESSION_ID', session_id)
        langfuse_user_id = os.getenv('LANGFUSE_USER', 'default-user')
        # Skip environment validation for Gemini models
        if 'gemini' in model_name:
            # No environment keys needed for these
            pass
        else:
            env_val = validate_environment(model_name)
            # Only raise an error if there are actually missing keys
            if not env_val['keys_in_environment'] and env_val['missing_keys']:
                raise ValueError(f"Missing {env_val['missing_keys']} in the environment.")
                
        self.model_name, provider, _, _ = litellm_utils.get_llm_provider(model_name)
        self.model = model_name
        # Persist provider so we can gate unsupported params (e.g., Gemini penalties)
        self.provider = provider
        
        # Skip key validation for models that don't need API keys
        # if 'gemini' not in model_name and not self.check_valid_key(model_name):
        #     raise ValueError(f"Invalid keys for model {model_name}.")
        
        self.generation_config = generation_config or GenerativeModelConfig()
        # If the caller did not provide an explicit max_tokens value, set a conservative default
        # to avoid hitting provider hard limits (Anthropic = 4 096, OpenAI = 8 192, Gemini similar).
        # A lower limit (1 000) still allows sizeable responses but keeps us clear of the
        # "stop_reason=max_tokens" error and reduces cost.
        if self.generation_config.max_tokens is None:
            self.generation_config.max_tokens = 8192
        self.system_instruction = system_instruction
        
        # Set API key if provided
        if api_key:
            self.kwargs = {"api_key": api_key}
            self.kwargs.update(kwargs)
        else:
            self.kwargs = kwargs

    def check_valid_key(self, model: str):
        """
        Checks if the environment key is valid for a specific model by making a litellm.completion call with max_tokens=10

        Args:
            model (str): The name of the model to check the key against.

        Returns:
            bool: True if the key is valid for the model, False otherwise.
        """
        messages = [{"role": "user", "content": "Hey, how's it going?"}]
        try:
            completion(
                model=model, messages=messages, max_tokens=10
            )
            return True
        except Exception as e:
            return False

    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        """
        Start a new chat session.
        
        Args:
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
            
        Returns:
            GenerativeModelChatSession: A new instance of the chat session.
        """
        return LiteModelChatSession(self, system_instruction)

    @observe(name="GenerateResponseLiteLLM",  as_type="generation")
    def generate(self, prompt: str) -> str:
        """
        Generate a response for a single prompt.
        
        Args:
            prompt (str): The prompt to generate a response for.
            
        Returns:
            str: The generated response.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id)
        response = self.ask(prompt)
        return response.text

    @observe(name="GenerateResponseWithDetailsLiteLLM",  as_type="generation")
    def generate_with_details(self, prompt: str) -> GenerationResponse:
        """
        Generate a response with additional details.
        
        Args:
            prompt (str): The prompt to generate a response for.
            
        Returns:
            GenerationResponse: The generated response with additional details.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id)
        return self.ask(prompt)

    @observe(name="AskLiteLLM",  as_type="generation")
    def ask(self, message: str) -> GenerationResponse:
        """
        Send a message to the model and receive a response.
        
        Args:
            message (str): The message to send.
            
        Returns:
            GenerationResponse: The model's generated response.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id)
        messages = [{"role": "user", "content": message[:32000]}]
        
        # Only include system message if provided
        # if self.system_instruction:
            #messages.insert(0, {"role": "system", "content": self.system_instruction}) todos
            #messages.insert(0, {"role": "user", "content": self.system_instruction}) todos

        try:
            # Build params, omitting unsupported fields for Gemini
            params: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }

            if self.generation_config.temperature is not None:
                params["temperature"] = self.generation_config.temperature
            if self.generation_config.max_tokens is not None:
                params["max_tokens"] = self.generation_config.max_tokens
            if self.generation_config.top_p is not None:
                params["top_p"] = self.generation_config.top_p

            # Do not forward penalty params to Gemini providers
            is_gemini = ("gemini" in (self.model or "").lower()) or (getattr(self, "provider", "").lower() == "gemini")
            if not is_gemini:
                if self.generation_config.frequency_penalty is not None:
                    params["frequency_penalty"] = self.generation_config.frequency_penalty
                if self.generation_config.presence_penalty is not None:
                    params["presence_penalty"] = self.generation_config.presence_penalty

            response = completion(**params)
            parsed_response = self._parse_generate_content_response(response)
            self.add_ask_to_langfuse(response, parsed_response)
            return parsed_response
        except Exception as e:
            raise ValueError(f"Error during generation, check credentials - {e}") from e

    def _parse_generate_content_response(self, response: Any) -> GenerationResponse:
        """
        Parse the model's response and extract content for the user.
        
        Args:
            response (Any): The raw response from the model.
            
        Returns:
            GenerationResponse: Parsed response containing the generated text.
        """
        if not response or not response.get("choices") or not response["choices"]:
            return GenerationResponse(
                text="No response generated", finish_reason=FinishReason.OTHER
            )
        
        choice = response["choices"][0]
        text = choice.get("message", {}).get("content", "") or ""
        
        # Parse the finish reason
        if "finish_reason" in choice:
            if choice["finish_reason"] == "stop":
                finish_reason = FinishReason.STOP
            elif choice["finish_reason"] == "length":
                finish_reason = FinishReason.MAX_TOKENS
            else:
                finish_reason = FinishReason.OTHER
        else:
            finish_reason = FinishReason.STOP
        
        return GenerationResponse(text=text, finish_reason=finish_reason)

    def parse_generate_content_response(self, response: Any) -> GenerationResponse:
        """
        Parse the model's response and extract content for the user.
        
        Args:
            response (Any): The raw response from the model.
            
        Returns:
            GenerationResponse: Parsed response containing the generated text.
        """
        return self._parse_generate_content_response(response)

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize the model's configuration and state to JSON format.
        
        Returns:
            Dict[str, Any]: The serialized JSON data.
        """
        return {
            "model_name": self.model,
            "generation_config": self.generation_config.to_json(),
            "system_instruction": self.system_instruction,
        }

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "LiteModel":
        """
        Deserialize a JSON object to create an instance of LiteModel.
        
        Args:
            json (Dict[str, Any]): The serialized JSON data.
            
        Returns:
            LiteModel: A new instance of the model.
        """
        return LiteModel(
            model_name=json["model_name"],
            generation_config=GenerativeModelConfig.from_json(
                json["generation_config"]
            ),
            system_instruction=json["system_instruction"],
        )
    
    def add_ask_to_langfuse(self, response, parsed_response):
        """
        Add the ask messages to Langfuse for tracing.
        
        Args:
            response (Any): The raw response from the model.
            parsed_response (GenerationResponse): The parsed response containing the generated text.
        """
        try:
            usage = response.usage.model_dump()
        except Exception as e:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "message": f"Error during response.dump in result - {str(e)}"}
        #if parased_response is a text, take it as it is, otherwise use parsed_response.text if .text exists
        if isinstance(parsed_response, str):
            output_text = parsed_response
        else:
            output_text = parsed_response.text if hasattr(parsed_response, 'text') else str(parsed_response)
        # Extract and clean the model name, removing any 'gemini/' prefix
        # This is necessary to ensure Langfuse identifies the model correctly, and traces it properly for cost and usage
        model_name = self.model
        if model_name.startswith("gemini/"):
            model_name_lang = model_name.split("/")[1]  # Extract the model name after 'gemini/'
        else:
            model_name_lang = model_name
        try:
            langfuse.update_current_generation(output= output_text)
            langfuse.update_current_generation(model = model_name_lang)
            langfuse.update_current_generation(usage_details={
                                                "input": usage.get("prompt_tokens", 0),
                                                "output": usage.get("completion_tokens", 0),
                                                "total": usage.get("total_tokens", 0),
                                                            })
            try:
                cost = f"${completion_cost(response):.8f}"
            except Exception as e:
                cost = "X.XXXXX"
            langfuse.update_current_generation(metadata={
                                                    "usage": usage,
                                                    "cost": cost,
                                                    "response_time" : getattr(response, "_response_ms", None),
                                                })
        except Exception as e:
            langfuse.update_current_generation(metadata={
                                                    "error_msg": f"Error during Langfuse update - {str(e)}",
                                                })
    
class LiteModelChatSession(GenerativeModelChatSession):
    """
    A chat session for interacting with the LiteLLM, maintaining conversation history.
    """
    
    def __init__(self, model: LiteModel, system_instruction: Optional[str] = None):
        """
        Initialize the chat session and set up the conversation history.
        
        Args:
            model (LiteModel): The model instance for the session.
            system_instruction (Optional[str]): Optional system instruction to guide the chat session.
        """
        self._model = model
        self._system_instruction = system_instruction or model.system_instruction
        
        # Simple message history list
        self._messages: List[Dict[str, str]] = []
        
        # Add the system message if provided
        if self._system_instruction:
            self._messages.append({"role": "system", "content": self._system_instruction})
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Retrieve the conversation history for the current chat session.
        
        Returns:
            List[Dict[str, str]]: The chat session's conversation history.
        """
        return self._messages
    
    @observe(name="send_messageLiteLLM", as_type="generation")
    def send_message(self, message: str, output_method: OutputMethod = OutputMethod.DEFAULT) -> GenerationResponse:
        """
        Send a message in the chat session and receive the model's response.
        
        Args:
            message (str): The message to send.
            output_method (OutputMethod): Format for the model's output.
            
        Returns:
            GenerationResponse: The generated response.
        """
        langfuse.update_current_trace(session_id=session_id, user_id=langfuse_user_id)
        # Add user message to history
        self._messages.append({"role": "user", "content": message[:32000]})
        
        # Adjust generation config based on output method
        generation_config = self._adjust_generation_config(output_method)
        
        # Simple retry mechanism for model overload errors
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Call the LiteLLM completion function with message history
                response = completion(
                    model=self._model.model,
                    messages=self._messages,
                    **generation_config
                )
                
                # Parse response and add to history
                content = self._model._parse_generate_content_response(response)
                self._messages.append({"role": "assistant", "content": content.text})
                self.add_chat_to_langfuse(response, content)
                return content
            except Exception as e:
                error_msg = str(e).lower()
                if attempt < max_retries and ('overloaded' in error_msg or '503' in error_msg or 'unavailable' in error_msg):
                    # Wait a bit before retrying
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    continue
                else:
                    raise ValueError(f"Error during generation, check credentials - {e}") from e
    
    def _adjust_generation_config(self, output_method: OutputMethod) -> Dict[str, Any]:
        """
        Adjust the generation configuration based on the output method.
        
        Args:
            output_method (OutputMethod): The desired output method (e.g., default or JSON).
            
        Returns:
            Dict[str, Any]: The adjusted configuration settings for generation.
        """
        # Build config, include only non-None values
        cfg_src = self._model.generation_config
        config: Dict[str, Any] = {}
        if cfg_src.temperature is not None:
            config["temperature"] = cfg_src.temperature
        if cfg_src.max_tokens is not None:
            config["max_tokens"] = cfg_src.max_tokens
        if cfg_src.top_p is not None:
            config["top_p"] = cfg_src.top_p

        # Some providers (e.g., Gemini via Vertex) do not support presence/frequency penalties
        is_gemini = ("gemini" in (self._model.model or "").lower()) or (getattr(self._model, "provider", "").lower() == "gemini")
        if not is_gemini:
            if cfg_src.frequency_penalty is not None:
                config["frequency_penalty"] = cfg_src.frequency_penalty
            if cfg_src.presence_penalty is not None:
                config["presence_penalty"] = cfg_src.presence_penalty
        
        # For JSON output, adjust temperature and set response format
        if output_method == OutputMethod.JSON:
            config["temperature"] = 0.0
            config["response_format"] = {"type": "json_object"}
        
        return config
    
    
    def delete_last_message(self):
        """
        Deletes the last message exchange (user message and assistant response) from the context.
        Preserves the system message if present.
        """
        # Remove the last two messages (user + assistant) if they exist
        if len(self._messages) >= 3:  # system + user + assistant
            # Remove last two messages (user and assistant)
            self._messages = self._messages[:-2]
        elif len(self._messages) == 2:  # system + user (no assistant response yet)
            # Remove just the user message
            self._messages = self._messages[:-1]
        # If only system message or empty, do nothing
    
    def get_context_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current context window.
        
        Returns:
            Dict[str, Any]: Context statistics including message count, etc.
        """
        return {
            "total_messages": len(self._messages),
            "has_system_message": any(msg.get("role") == "system" for msg in self._messages)
        }
    
    def clear_context(self):
        """
        Clear all conversation context while preserving the system instruction.
        """
        self._messages = []
        if self._system_instruction:
            self._messages.append({"role": "system", "content": self._system_instruction})
    
    def add_chat_to_langfuse(self, response, parsed_response):
        """
        Add the chat messages to Langfuse for tracing.
        
        Args:
            response (Any): The raw response from the model.
            parsed_response (GenerationResponse): The parsed response containing the generated text.
        """
        try:
            usage = response.usage.model_dump()
        except Exception as e:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "message": f"Error during response.dump in result - {str(e)}"}
        #if parased_response is a text, take it as it is, otherwise use parsed_response.text if .text exists
        if isinstance(parsed_response, str):
            output_text = parsed_response
        else:
            output_text = parsed_response.text if hasattr(parsed_response, 'text') else str(parsed_response)
        # Extract and clean the model name, removing any 'gemini/' prefix
        # This is necessary to ensure Langfuse identifies the model correctly, and traces it properly for cost and usage
        model_name = self._model.model_name
        if model_name.startswith("gemini/"):
            model_name_lang = model_name.split("/")[1]  # Extract the model name after 'gemini/'
        else:
            model_name_lang = model_name
        try:
            langfuse.update_current_generation(output= output_text)
            langfuse.update_current_generation(model = model_name_lang)
            langfuse.update_current_generation(usage_details={
                                                "input": usage.get("prompt_tokens", 0),
                                                "output": usage.get("completion_tokens", 0),
                                                "total": usage.get("total_tokens", 0),
                                                            })
            try:
                cost = f"${completion_cost(response):.8f}"
            except Exception as e:
                cost = "X.XXXXX"
            langfuse.update_current_generation(metadata={
                                                    "usage": usage,
                                                    "cost": cost,
                                                    "response_time" : getattr(response, "_response_ms", None),
                                                })
        except Exception as e:
            langfuse.update_current_generation(metadata={
                                                    "error_msg": f"Error during Langfuse update - {str(e)}",
                                                })

    