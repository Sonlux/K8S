from langchain.agents import Tool
from langchain.llms.base import LLM
from typing import List, Dict, Any
import logging

class BaseAgent:
    """Base class for all specialized agents in the Multi-Agent System"""
    
    def __init__(self, llm: LLM):
        """Initialize the base agent with an LLM"""
        self.llm = llm
        self.logger = logging.getLogger(f"mas-agent-{self.__class__.__name__.lower()}")
    
    def get_tools(self) -> List[Tool]:
        """Get the list of tools available to this agent"""
        raise NotImplementedError("Subclasses must implement get_tools()")
    
    def process_result(self, result: str) -> Dict[str, Any]:
        """Process the result from an agent's action"""
        raise NotImplementedError("Subclasses must implement process_result()")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for the agent"""
        raise NotImplementedError("Subclasses must implement validate_input()")
    
    def format_output(self, output_data: Dict[str, Any]) -> str:
        """Format the output data from the agent"""
        raise NotImplementedError("Subclasses must implement format_output()")
    
    def log_action(self, action: str, details: Dict[str, Any]) -> None:
        """Log an action taken by the agent"""
        self.logger.info(f"Action: {action}, Details: {details}")
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors in agent operations"""
        self.logger.error(f"Error: {str(error)}")
        return {
            "success": False,
            "error": str(error)
        } 