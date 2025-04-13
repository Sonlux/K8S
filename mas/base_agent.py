from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, List
from kubernetes import client
import time

class BaseAgent(ABC):
    def __init__(self, name: str, k8s_api: client.CoreV1Api, k8s_apps_api: client.AppsV1Api):
        self.name = name
        self.k8s_api = k8s_api
        self.k8s_apps_api = k8s_apps_api
        self.logger = logging.getLogger(f"mas-agent-{name}")
        self.confidence_threshold = 0.8
        self.action_history: List[Dict[str, Any]] = []

    @abstractmethod
    def can_handle(self, metrics: Dict[str, Any]) -> bool:
        """Determine if this agent can handle the given metrics"""
        pass

    @abstractmethod
    def analyze(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics and return analysis results"""
        pass

    @abstractmethod
    def remediate(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Take remediation action based on analysis"""
        pass

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