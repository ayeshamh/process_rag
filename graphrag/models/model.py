from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, Iterator


class FinishReason:
    MAX_TOKENS = "MAX_TOKENS"
    STOP = "STOP"
    OTHER = "OTHER"

class OutputMethod(Enum):
    JSON = 'json'
    DEFAULT = 'default'

class GenerativeModelConfig:
    """
    Configuration for a generative model.
    
    This configuration follows OpenAI-style parameter naming but is designed to be compatible with other generative models.
    
    Args:
        temperature (Optional[float]): Controls the randomness of the output. Higher values make responses more random,
            while lower values make them more deterministic.
        top_p (Optional[float]): Nucleus sampling parameter. A value of 0.9 considers only the top 90% of probability mass.
        top_k (Optional[int]): Limits sampling to the top-k most probable tokens.
        max_tokens (Optional[int]): The maximum number of tokens the model is allowed to generate in a response.
        stop (Optional[list[str]]): A list of stop sequences that signal the model to stop generating further tokens.
        response_format (Optional[dict]): Specifies the desired format of the response, if supported by the model.
        frequency_penalty (Optional[float]): The frequency penalty to apply to the model's responses.
        presence_penalty (Optional[float]): The presence penalty to apply to the model's responses.
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        response_format: Optional[dict] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.stop = stop
        self.response_format = response_format
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def __str__(self) -> str:
        return f"GenerativeModelConfig(temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, max_tokens={self.max_tokens}, stop={self.stop})"

    def to_json(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "response_format": self.response_format,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    @staticmethod
    def from_json(json: dict) -> "GenerativeModelConfig":
        return GenerativeModelConfig(
            temperature=json.get("temperature"),
            top_p=json.get("top_p"),
            top_k=json.get("top_k"),
            max_tokens=json.get("max_tokens"),
            stop=json.get("stop"),
            response_format=json.get("response_format"),
            frequency_penalty=json.get("frequency_penalty"),
            presence_penalty=json.get("presence_penalty"),
        )


class GenerationResponse:
    """
    Response from a generative model.
    
    Args:
        text (str): The generated text.
        finish_reason (str): The reason the generation stopped.
    """

    def __init__(self, text: str, finish_reason: str):
        self.text = text
        self.finish_reason = finish_reason


class GenerativeModelChatSession(ABC):
    """
    Abstract class for a generative model chat session.
    """

    @abstractmethod
    def send_message(self, message: str, output_method: OutputMethod = OutputMethod.DEFAULT) -> GenerationResponse:
        """
        Send a message to the model and get a response.
        
        Args:
            message (str): The message to send.
            output_method (OutputMethod): The output method to use.
            
        Returns:
            GenerationResponse: The model's response.
        """
        pass
    
    
    @abstractmethod
    def get_chat_history(self) -> list[dict]:
        """
        Get the chat history.
        
        Returns:
            list[dict]: The chat history.
        """
        pass
    
    @abstractmethod
    def delete_last_message(self):
        """
        Delete the last message exchange from chat history.
        """
        pass


class GenerativeModel(ABC):
    """
    Abstract class for a generative model.
    """

    @abstractmethod
    def start_chat(self, system_instruction: Optional[str] = None) -> GenerativeModelChatSession:
        """
        Start a chat session with the model.
        
        Args:
            system_instruction (Optional[str]): System instruction to guide the model.
            
        Returns:
            GenerativeModelChatSession: A chat session with the model.
        """
        pass

    @abstractmethod
    def parse_generate_content_response(self, response: any) -> GenerationResponse:
        """
        Parse a response from the model.
        
        Args:
            response (any): The response to parse.
            
        Returns:
            GenerationResponse: The parsed response.
        """
        pass

    @abstractmethod
    def to_json(self) -> dict:
        """
        Convert the model to a JSON representation.
        
        Returns:
            dict: A JSON representation of the model.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_json(json: dict) -> "GenerativeModel":
        """
        Create a model from a JSON representation.
        
        Args:
            json (dict): A JSON representation of the model.
            
        Returns:
            GenerativeModel: A model created from the JSON representation.
        """
        pass
