class BaseStep:
    """
    Base class for all steps.
    This serves as a common interface for step implementations.
    """
    
    def __init__(self, **kwargs):
        """Initialize the step with the given arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self, *args, **kwargs):
        """
        Run the step.
        This method should be implemented by subclasses.
        
        Returns:
            Any: The result of running the step.
        """
        raise NotImplementedError("Subclasses must implement the run method.") 