#!/usr/bin/env python3

import os
import sys
import logging
from dotenv import load_dotenv
from langchain.llms.base import LLM
from typing import Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockLLM(LLM):
    """A mock LLM for testing purposes."""
    
    def __init__(self):
        """Initialize the mock LLM."""
        super().__init__()
        logger.info("Initializing MockLLM")
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generate a mock response based on the prompt.
        
        Args:
            prompt: The input prompt
            stop: Optional list of strings to stop generation
            
        Returns:
            A mock response based on the prompt content
        """
        logger.info(f"MockLLM received prompt: {prompt[:100]}...")
        
        # Generate mock responses based on prompt content
        if "analyze" in prompt.lower():
            return "Based on the metrics, I've identified potential resource issues in pod 'test-pod'. CPU usage is at 85% and memory usage is at 90%."
        elif "remediate" in prompt.lower():
            return "I recommend scaling up the pod's resources. CPU limit should be increased by 50% and memory limit by 25%."
        elif "monitor" in prompt.lower():
            return "Monitoring metrics for pod 'test-pod'. Current status: Running, CPU: 75%, Memory: 80%"
        else:
            return "I understand the request and will process it accordingly."
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "mock"

def main():
    """Run the Multi-Agent System with a mock LLM."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Create mock LLM
        llm = MockLLM()
        
        # Import and run the MAS
        from mas.langgraph_system import run_mas
        run_mas(llm=llm)
        
    except Exception as e:
        logger.error(f"Error running MAS with mock LLM: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 