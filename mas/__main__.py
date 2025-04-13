#!/usr/bin/env python3
import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
import requests
import json
from pydantic import Field, PrivateAttr

# Import our custom modules
from .agents.monitoring_agent import MonitoringAgent
from .agents.remediation_agent import RemediationAgent
from .agents.analysis_agent import AnalysisAgent
from .agents.resource_agent import ResourceAgent
from .agents.network_agent import NetworkAgent
from .agents.log_agent import LogAgent
from .coordinator import Coordinator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaLLM(LLM):
    """Llama LLM implementation using the Llama API."""
    
    model: str = Field(default="llama-3.1-8b-instant")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    
    _api_key: str = PrivateAttr(default="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")
    _base_url: str = PrivateAttr(default="https://api.llamaapi.net/v1")
    
    def __init__(self, **kwargs):
        """Initialize the Llama LLM."""
        super().__init__(**kwargs)
        logger.info(f"Initializing LlamaLLM with model: {self.model}")
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Call the Llama API with the given prompt.
        
        Args:
            prompt: The input prompt
            stop: Optional list of strings to stop generation
            
        Returns:
            The generated text
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if stop:
            data["stop"] = stop
        
        try:
            response = requests.post(
                f"{self._base_url}/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Error calling Llama API: {e}")
            raise
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llama"

def setup_environment() -> bool:
    """Setup the environment for the MAS"""
    required_vars = [
        "LLAMA_API_KEY",
        "KUBERNETES_CONFIG_PATH",
        "PROMETHEUS_URL"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file")
        return False
    
    # Expand the Kubernetes config path if it contains ~
    k8s_config = os.getenv("KUBERNETES_CONFIG_PATH")
    if k8s_config and k8s_config.startswith("~"):
        os.environ["KUBERNETES_CONFIG_PATH"] = os.path.expanduser(k8s_config)
    
    return True

def create_llm():
    """Create and return a LlamaLLM instance"""
    return LlamaLLM()

def initialize_agents(llm: LLM) -> List[Any]:
    """Initialize all agents with the provided LLM"""
    agents = []
    
    # Initialize each agent with the LLM
    agents.append(MonitoringAgent(llm=llm))
    agents.append(RemediationAgent(llm=llm))
    agents.append(AnalysisAgent(llm=llm))
    agents.append(ResourceAgent(llm=llm))
    agents.append(NetworkAgent(llm=llm))
    agents.append(LogAgent(llm=llm))
    
    return agents

def create_coordinator(agents: List[Any], llm: LLM) -> Coordinator:
    """Create the coordinator with the initialized agents"""
    # Create the memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=int(os.getenv("AGENT_MEMORY_SIZE", "1000"))
    )
    
    # Create the system message
    system_message = SystemMessage(
        content="""You are a Kubernetes cluster management assistant.
        Your job is to help monitor and manage the cluster, identify issues,
        and suggest remediation actions. You have access to various tools
        through the agents that can help you accomplish this task."""
    )
    
    # Create the prompt
    prompt = [
        system_message,
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
    
    # Collect tools from all agents
    tools = []
    for agent in agents:
        tools.extend(agent.get_tools())
    
    # Create the coordinator
    return Coordinator(llm, tools, prompt, memory)

def main():
    """Run the Multi-Agent System."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Create Llama LLM instance
        llm = LlamaLLM()
        
        # Initialize agents
        agents = initialize_agents(llm)
        
        # Create coordinator
        coordinator = create_coordinator(agents, llm)
        
        # Start monitoring loop
        monitoring_interval = int(os.getenv("MONITORING_INTERVAL", "60"))
        logger.info(f"Starting monitoring loop with interval of {monitoring_interval} seconds...")
        
        try:
            while True:
                # Monitor the cluster
                result = coordinator.monitor_cluster()
                
                # Log the result
                if "error" in result:
                    logger.error(f"Error monitoring cluster: {result['error']}")
                else:
                    logger.info(f"Processed {result['pods_processed']} pods, took {result['actions_taken']} actions")
                    
                    # Log any actions taken
                    for action in result.get("results", []):
                        logger.info(f"Action taken for pod {action['pod_name']} in namespace {action['namespace']}: {action['result']}")
                
                # Sleep for the monitoring interval
                time.sleep(monitoring_interval)
                
        except KeyboardInterrupt:
            logger.info("Shutting down Kubernetes Multi-Agent System...")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Error running MAS: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 