from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from kubernetes import client
import logging
import time
import os
from dotenv import load_dotenv

load_dotenv()

class LangChainBaseAgent:
    def __init__(self, name: str, k8s_api: client.CoreV1Api, k8s_apps_api: client.AppsV1Api):
        self.name = name
        self.k8s_api = k8s_api
        self.k8s_apps_api = k8s_apps_api
        self.logger = logging.getLogger(f"mas-agent-{name}")
        self.action_history: List[Dict[str, Any]] = []
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create tools for the agent
        self.tools = self._create_tools()
        
        # Create the agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        return f"""You are a specialized Kubernetes agent named {self.name} that helps monitor and remediate issues in a Kubernetes cluster.
Your role is to analyze metrics and take appropriate actions to maintain cluster health.

You have access to the following tools:
{self._get_tools_description()}

When analyzing metrics:
1. First determine if the metrics indicate an issue that needs attention
2. If an issue is detected, analyze its severity and potential impact
3. Choose the most appropriate action to remediate the issue
4. Execute the chosen action and monitor its effectiveness

Always be cautious and:
- Don't take actions unless you're confident they're needed
- Consider the impact of your actions on the cluster
- Document your reasoning for taking any action
- Learn from the results of your actions

Current metrics and cluster state will be provided to you in a structured format."""

    def _get_tools_description(self) -> str:
        """Get descriptions of available tools"""
        return "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])

    def _create_tools(self) -> List[BaseTool]:
        """Create the tools available to the agent"""
        raise NotImplementedError("Subclasses must implement _create_tools")

    def analyze_and_act(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics and take appropriate action"""
        try:
            # Format metrics for the agent
            input_text = self._format_metrics_for_agent(metrics)
            
            # Get agent's analysis and action
            result = self.agent_executor.invoke({
                "input": input_text,
                "chat_history": []
            })
            
            # Parse and record the action
            action_result = self._parse_agent_result(result)
            self.record_action(
                action_result.get('action', 'unknown'),
                action_result.get('success', False),
                action_result.get('details', 'No details provided')
            )
            
            return action_result
            
        except Exception as e:
            self.logger.error(f"Error in agent execution: {str(e)}")
            return {'action_taken': False, 'error': str(e)}

    def _format_metrics_for_agent(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a natural language description"""
        return f"""Please analyze the following Kubernetes metrics and determine if any action is needed:

Metrics:
{self._format_metrics(metrics)}

Please analyze these metrics and determine:
1. If there are any issues that need attention
2. The severity of any issues found
3. What action, if any, should be taken
4. The expected impact of the action

Respond with your analysis and recommended action."""

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a readable string"""
        return "\n".join([f"- {k}: {v}" for k, v in metrics.items()])

    def _parse_agent_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the agent's result into a structured format"""
        # This should be implemented by subclasses to parse the specific
        # format of results they expect from their agents
        raise NotImplementedError("Subclasses must implement _parse_agent_result")

    def record_action(self, action: str, success: bool, details: str):
        """Record an action taken by the agent"""
        self.action_history.append({
            'action': action,
            'success': success,
            'details': details,
            'timestamp': time.time()
        })

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get the history of actions taken by this agent"""
        return self.action_history

    def reset_history(self):
        """Reset the action history"""
        self.action_history = [] 