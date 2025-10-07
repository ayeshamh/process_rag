from graphrag.models import GenerativeModel


class KnowledgeGraphModelConfig:
    """
    A model configuration for knowledge graph operations.
    """

    def __init__(
        self,
        qa: GenerativeModel = None,
    ) -> None:
        """
        Creates a model configuration object that wraps the models used for knowledge graph operations.

        Args:
            qa (AbstractGenerativeModel): The model to use for question answering.
        """
        self.qa = qa

    @staticmethod
    def with_model(model: GenerativeModel):
        """
        Create a configuration using the specified model for all operations.

        Args:
            model (GenerativeModel): The model to use for all operations.

        Returns:
            KnowledgeGraphModelConfig: A new configuration using the specified model.
        """
        return KnowledgeGraphModelConfig(qa=model)

    def to_json(self) -> dict:
        """
        Convert the configuration to a JSON-serializable object.

        Returns:
            dict: The configuration as a JSON object.
        """
        return {
            "qa": self.qa.to_json() if self.qa else None,
        }

    @staticmethod
    def from_json(json: dict, qa_model: GenerativeModel = None) -> "KnowledgeGraphModelConfig":
        """
        Create a configuration from a JSON object.

        Args:
            json (dict): The JSON object.
            qa_model (GenerativeModel): A model to use for question answering.

        Returns:
            KnowledgeGraphModelConfig: A new configuration.
        """
        return KnowledgeGraphModelConfig(
            qa=qa_model,
        )
