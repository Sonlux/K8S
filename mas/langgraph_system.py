#!/usr/bin/env python3

import os
import time
import logging
import json
import re
from typing import Dict, List, Tuple, Any, TypedDict, Annotated, Optional
from datetime import datetime
from dataclasses import dataclass, field
from kubernetes import client, config
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from langchain_community.llms import LlamaCpp
from dotenv import load_dotenv

# Try to import LangGraph, but provide a fallback to LangChain
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not available, falling back to LangChain")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions for our state
class AgentState(TypedDict):
    """State for each agent in the system."""
    messages: List[Tuple[str, str]]  # List of (role, content) message tuples
    next_agent: str  # Next agent to execute
    current_agent: str  # Current agent being executed
    metrics: Dict[str, Any]  # Current metrics data
    remediation_history: Dict[str, List[Dict[str, Any]]]  # History of remediation actions
    last_remediation: Dict[str, float]  # Timestamp of last remediation per pod
    issues: List[Dict[str, Any]]  # List of identified issues

class KubernetesState(TypedDict):
    """State for Kubernetes operations."""
    api_client: Any  # Kubernetes API client
    namespace: str  # Current namespace
    simulation_mode: bool  # Whether we're in simulation mode

@dataclass
class MetricsData:
    """Data class for pod metrics."""
    pod_id: str
    namespace: str
    pod_name: str
    cpu_usage: float
    memory_usage: float
    network_errors: int
    network_packets_dropped: int
    timestamp: datetime = field(default_factory=datetime.now)

class KubernetesTools:
    """Tools for interacting with Kubernetes."""
    
    def __init__(self, state: KubernetesState):
        self.state = state
        self.api_client = state["api_client"]
        self.namespace = state["namespace"]
        self.simulation_mode = state["simulation_mode"]
    
    @tool
    def get_pod_metrics(self, pod_id: str) -> Dict[str, Any]:
        """Get metrics for a specific pod."""
        if self.simulation_mode:
            return {
                "cpu_usage": 0.5,
                "memory_usage": 0.6,
                "network_errors": 0,
                "network_packets_dropped": 0
            }
        
        try:
            # In a real implementation, this would fetch actual metrics
            # For now, we'll return simulated data
            return {
                "cpu_usage": 0.5,
                "memory_usage": 0.6,
                "network_errors": 0,
                "network_packets_dropped": 0
            }
        except Exception as e:
            logger.error(f"Error getting metrics for pod {pod_id}: {str(e)}")
            return {}
    
    @tool
    def get_pod_logs(self, pod_id: str) -> str:
        """Get logs for a specific pod."""
        if self.simulation_mode:
            return "Simulated pod logs"
        
        try:
            # In a real implementation, this would fetch actual logs
            return "Pod logs"
        except Exception as e:
            logger.error(f"Error getting logs for pod {pod_id}: {str(e)}")
            return ""
    
    @tool
    def restart_pod(self, pod_id: str) -> bool:
        """Restart a specific pod."""
        if self.simulation_mode:
            logger.info(f"Simulating restart of pod {pod_id}")
            return True
        
        try:
            # In a real implementation, this would restart the pod
            logger.info(f"Restarting pod {pod_id}")
            return True
        except Exception as e:
            logger.error(f"Error restarting pod {pod_id}: {str(e)}")
            return False

class MonitoringAgent:
    """Agent responsible for monitoring pod metrics."""
    
    def __init__(self, llm: LlamaCpp):
        self.llm = llm
        self.tools = None  # Will be set when tools are available
    
    def setup_tools(self, tools: KubernetesTools):
        """Set up the tools for this agent."""
        self.tools = tools
    
    def process(self, state: AgentState) -> AgentState:
        """Process the current state and update metrics."""
        logger.info("Monitoring agent processing state")
        
        # Get metrics for all pods
        metrics = {}
        for pod_id in state["metrics"]:
            metrics[pod_id] = self.tools.get_pod_metrics(pod_id)
        
        # Update state with new metrics
        state["metrics"] = metrics
        state["next_agent"] = "analysis"
        
        return state

class AnalysisAgent:
    """Agent responsible for analyzing pod metrics and logs with ML-based anomaly detection."""
    
    def __init__(self, llm: LlamaCpp):
        self.llm = llm
        self.tools = None
        self.metric_history = {}
        self.anomaly_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 80.0,
            'restart_count': 3,
            'network_errors': 5
        }
        self.correlation_window = 300  # 5 minutes
        self.training_data = []
    
    def setup_tools(self, tools: KubernetesTools):
        """Set up the tools for this agent."""
        self.tools = tools
    
    def process(self, state: AgentState) -> AgentState:
        """Analyze pod metrics and logs with ML-based anomaly detection."""
        logger.info("Analysis agent processing state")
        
        current_time = time.time()
        for pod_id, metrics in state["pod_metrics"].items():
            # Update metric history
            if pod_id not in self.metric_history:
                self.metric_history[pod_id] = []
            self.metric_history[pod_id].append({
                'timestamp': current_time,
                'metrics': metrics
            })
            
            # Trim history to correlation window
            self._trim_history(pod_id)
            
            # Detect anomalies using ML
            anomalies = self._detect_anomalies(pod_id, metrics)
            
            # Correlate with other pods
            correlations = self._correlate_with_other_pods(pod_id, anomalies)
            
            # Generate analysis using LLM
            analysis = self._generate_analysis(pod_id, metrics, anomalies, correlations)
            
            # Update state with analysis
            if analysis:
                state["analysis_results"][pod_id] = analysis
                state["next_agent"] = "remediation"
        
        return state
    
    def _trim_history(self, pod_id: str):
        """Trim metric history to correlation window."""
        current_time = time.time()
        self.metric_history[pod_id] = [
            entry for entry in self.metric_history[pod_id]
            if current_time - entry['timestamp'] <= self.correlation_window
        ]
    
    def _detect_anomalies(self, pod_id: str, current_metrics: Dict) -> List[Dict]:
        """Detect anomalies using ML-based approach."""
        anomalies = []
        
        # Get historical metrics
        history = self.metric_history[pod_id]
        if len(history) < 2:
            return anomalies
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m['metrics'].get('CPU Usage (%)', 0) for m in history])
        memory_trend = self._calculate_trend([m['metrics'].get('Memory Usage (%)', 0) for m in history])
        
        # Check for sudden spikes
        if cpu_trend > 0.5:  # 50% increase
            anomalies.append({
                'type': 'resource_exhaustion',
                'metric': 'CPU Usage',
                'severity': 'high',
                'trend': cpu_trend
            })
        
        if memory_trend > 0.5:
            anomalies.append({
                'type': 'resource_exhaustion',
                'metric': 'Memory Usage',
                'severity': 'high',
                'trend': memory_trend
            })
        
        # Check for crash loops
        restart_count = current_metrics.get('Restart Count', 0)
        if restart_count >= self.anomaly_thresholds['restart_count']:
            anomalies.append({
                'type': 'crash_loop',
                'metric': 'Restart Count',
                'severity': 'high',
                'count': restart_count
            })
        
        # Check for network issues
        network_errors = current_metrics.get('Network Errors', 0)
        if network_errors >= self.anomaly_thresholds['network_errors']:
            anomalies.append({
                'type': 'network_issue',
                'metric': 'Network Errors',
                'severity': 'medium',
                'count': network_errors
            })
        
        # Add to training data
        self.training_data.append({
            'timestamp': time.time(),
            'pod_id': pod_id,
            'metrics': current_metrics,
            'anomalies': anomalies
        })
        
        return anomalies
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression."""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        y = values
        
        # Simple linear regression
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = sum((xi - mean_x) ** 2 for xi in x)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def _correlate_with_other_pods(self, pod_id: str, anomalies: List[Dict]) -> List[Dict]:
        """Correlate anomalies with other pods."""
        correlations = []
        
        if not anomalies:
            return correlations
        
        # Get pod info
        pod = self.tools.get_pod(pod_id)
        if not pod:
            return correlations
        
        # Get pods in same namespace
        namespace = pod.metadata.namespace
        pods = self.tools.get_pods(namespace)
        
        for other_pod in pods:
            if other_pod.metadata.name == pod.metadata.name:
                continue
            
            other_id = f"{other_pod.metadata.namespace}/{other_pod.metadata.name}"
            if other_id not in self.metric_history:
                continue
            
            # Check for similar anomalies
            other_metrics = self.metric_history[other_id][-1]['metrics']
            similarity = self._calculate_metric_similarity(
                self.metric_history[pod_id][-1]['metrics'],
                other_metrics
            )
            
            if similarity > 0.8:  # 80% similarity threshold
                correlations.append({
                    'pod_id': other_id,
                    'similarity': similarity,
                    'metrics': other_metrics
                })
        
        return correlations
    
    def _calculate_metric_similarity(self, metrics1: Dict, metrics2: Dict) -> float:
        """Calculate similarity between two sets of metrics."""
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        if not common_metrics:
            return 0.0
        
        similarities = []
        for metric in common_metrics:
            val1 = float(metrics1[metric])
            val2 = float(metrics2[metric])
            
            # Normalize values
            max_val = max(val1, val2)
            if max_val == 0:
                continue
            
            similarity = 1 - abs(val1 - val2) / max_val
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _generate_analysis(self, pod_id: str, metrics: Dict, anomalies: List[Dict], correlations: List[Dict]) -> Dict:
        """Generate analysis using LLM."""
        if not anomalies:
            return None
        
        # Prepare prompt for LLM
        prompt = f"""Analyze the following pod metrics and anomalies:

Pod ID: {pod_id}
Current Metrics:
{json.dumps(metrics, indent=2)}

Detected Anomalies:
{json.dumps(anomalies, indent=2)}

Correlated Pods:
{json.dumps(correlations, indent=2)}

Provide a detailed analysis including:
1. Root cause of anomalies
2. Impact on system
3. Recommended remediation actions
4. Prevention strategies
"""
        
        try:
            # Get LLM analysis
            response = self.llm._call(prompt)
            
            # Parse response
            analysis = {
                'timestamp': time.time(),
                'pod_id': pod_id,
                'anomalies': anomalies,
                'correlations': correlations,
                'llm_analysis': response,
                'recommended_actions': self._extract_recommended_actions(response)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return None
    
    def _extract_recommended_actions(self, llm_response: str) -> List[str]:
        """Extract recommended actions from LLM response."""
        actions = []
        lines = llm_response.split('\n')
        
        for line in lines:
            if line.strip().startswith('- '):
                actions.append(line.strip()[2:])
        
        return actions

class RemediationAgent:
    """Agent responsible for remediating identified issues with enhanced coordination."""
    
    def __init__(self, llm: LlamaCpp):
        self.llm = llm
        self.tools = None
        self.remediation_strategies = {
            'resource_exhaustion': self._handle_resource_exhaustion,
            'network_issue': self._handle_network_issue,
            'crash_loop': self._handle_crash_loop,
            'oom': self._handle_oom,
            'io_issue': self._handle_io_issue
        }
        self.state_history = {}
        self.rollback_points = {}
    
    def setup_tools(self, tools: KubernetesTools):
        """Set up the tools for this agent."""
        self.tools = tools
    
    def process(self, state: AgentState) -> AgentState:
        """Remediate identified issues with enhanced coordination."""
        logger.info("Remediation agent processing state")
        
        current_time = time.time()
        for issue in state["issues"]:
            pod_id = issue["pod_id"]
            
            # Check cooldown period
            if pod_id in state["last_remediation"]:
                time_since_last = current_time - state["last_remediation"][pod_id]
                if time_since_last < 300:  # 5-minute cooldown
                    logger.info(f"Pod {pod_id} in cooldown period")
                    continue
            
            # Create rollback point before remediation
            self._create_rollback_point(pod_id)
            
            # Get pod logs and metrics for diagnosis
            logs = self.tools.get_pod_logs(pod_id)
            metrics = self.tools.get_pod_metrics(pod_id)
            
            # Store current state
            self.state_history[pod_id] = {
                'timestamp': current_time,
                'metrics': metrics,
                'logs': logs,
                'issue': issue
            }
            
            # Select and execute remediation strategy
            strategy = self.remediation_strategies.get(issue["type"])
            if strategy:
                try:
                    result = strategy(pod_id, issue, metrics, logs)
                    if result.get('success'):
                        state["last_remediation"][pod_id] = current_time
                        self._update_remediation_history(state, pod_id, result)
                    else:
                        # Attempt rollback if remediation failed
                        self._rollback(pod_id)
                except Exception as e:
                    logger.error(f"Error in remediation strategy: {str(e)}")
                    self._rollback(pod_id)
            else:
                logger.warning(f"No remediation strategy found for issue type: {issue['type']}")
        
        return state

    def _handle_resource_exhaustion(self, pod_id: str, issue: Dict, metrics: Dict, logs: str) -> Dict:
        """Handle resource exhaustion with smart scaling."""
        try:
            # Get current resource usage
            cpu_usage = metrics.get('CPU Usage (%)', 0)
            memory_usage = metrics.get('Memory Usage (%)', 0)
            
            # Calculate scaling factor based on usage
            scale_factor = max(cpu_usage / 80.0, memory_usage / 80.0)
            
            # Get deployment info
            deployment = self.tools.get_deployment_for_pod(pod_id)
            if not deployment:
                return {'success': False, 'error': 'No deployment found'}
            
            # Check if HPA exists
            hpa = self.tools.get_hpa(deployment.metadata.name, deployment.metadata.namespace)
            if hpa:
                # Update HPA
                return self._update_hpa(hpa, scale_factor)
            else:
                # Scale deployment directly
                return self._scale_deployment(deployment, scale_factor)
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _handle_network_issue(self, pod_id: str, issue: Dict, metrics: Dict, logs: str) -> Dict:
        """Handle network issues with smart recovery."""
        try:
            # Check if it's a DNS issue
            if 'dns' in logs.lower():
                return self._fix_dns_issues(pod_id)
            
            # Check if it's a network policy issue
            if 'networkpolicy' in logs.lower():
                return self._fix_network_policy(pod_id)
            
            # Default to pod restart if no specific issue found
            return self._restart_pod(pod_id)
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _handle_crash_loop(self, pod_id: str, issue: Dict, metrics: Dict, logs: str) -> Dict:
        """Handle crash loops with sophisticated root cause analysis."""
        try:
            # Create rollback point before making changes
            self._create_rollback_point(pod_id)
            
            # Get pod details
            pod = self.tools.get_pod(pod_id)
            if not pod:
                return {'success': False, 'error': 'Pod not found'}
            
            # Get detailed container statuses
            container_statuses = pod.status.container_statuses
            if not container_statuses:
                return {'success': False, 'error': 'No container status available'}
            
            # Analyze container state and last state
            for status in container_statuses:
                if status.state.waiting:
                    reason = status.state.waiting.reason
                    message = status.state.waiting.message
                    
                    # Handle specific waiting states
                    if reason == 'CrashLoopBackOff':
                        # Check last terminated state for root cause
                        if status.last_state.terminated:
                            exit_code = status.last_state.terminated.exit_code
                            term_reason = status.last_state.terminated.reason
                            term_message = status.last_state.terminated.message
                            
                            # Handle different exit codes
                            if exit_code == 137 or term_reason == 'OOMKilled':
                                return self._handle_oom(pod_id, issue, metrics, logs)
                            elif exit_code == 1:
                                # Application error - analyze logs
                                return self._handle_application_error(pod_id, logs)
                            elif exit_code == 126:  # Permission denied
                                return self._fix_permissions(pod_id)
                            elif exit_code == 127:  # Command not found
                                return self._fix_command_configuration(pod_id, pod)
                            
                    elif reason == 'ImagePullBackOff':
                        return self._fix_image_pull_issues(pod_id, pod)
                    elif reason == 'RunContainerError':
                        return self._fix_container_configuration(pod_id, pod)
                    
            # Check for resource pressure
            node_name = pod.spec.node_name
            if node_name:
                node_metrics = self.tools.get_node_metrics(node_name)
                if self._is_node_pressured(node_metrics):
                    return self._handle_node_pressure(pod_id, node_name, node_metrics)
                
            # Check for network issues
            if self._has_network_issues(logs):
                return self._fix_network_issues(pod_id, pod)
            
            # If no specific issue found, try progressive remediation
            return self._apply_progressive_remediation(pod_id, pod)
            
        except Exception as e:
            # If remediation fails, attempt rollback
            if self._rollback(pod_id):
                return {
                    'success': False,
                    'error': f'Remediation failed: {str(e)}. Rolled back to previous state.'
                }
            return {'success': False, 'error': str(e)}

    def _handle_application_error(self, pod_id: str, logs: str) -> Dict:
        """Handle application-specific errors from logs."""
        try:
            # Extract error patterns from logs
            error_patterns = self._extract_error_patterns(logs)
            
            # Check for common application errors
            if any('database connection' in pattern.lower() for pattern in error_patterns):
                return self._fix_database_connection(pod_id)
            elif any('config' in pattern.lower() for pattern in error_patterns):
                return self._fix_config_issues(pod_id)
            elif any('dependency' in pattern.lower() for pattern in error_patterns):
                return self._fix_dependency_issues(pod_id)
            
            # Use LLM to analyze logs and suggest fixes
            analysis = self._analyze_logs_with_llm(logs)
            if analysis.get('fix_strategy'):
                return self._apply_llm_fix_strategy(pod_id, analysis['fix_strategy'])
            
            return {'success': False, 'error': 'Unable to determine application error cause'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _fix_command_configuration(self, pod_id: str, pod: Any) -> Dict:
        """Fix command and args configuration issues."""
        try:
            for container in pod.spec.containers:
                if container.command:
                    # Validate command exists in container
                    if not self._validate_command(pod_id, container.command[0]):
                        # Try to find correct command path
                        correct_path = self._find_correct_command_path(pod_id, container.command[0])
                        if correct_path:
                            container.command[0] = correct_path
                            self.tools.update_pod(pod)
                            return {
                                'success': True,
                                'action': 'fix_command_path',
                                'details': f'Updated command path to {correct_path}'
                            }
            return {'success': False, 'error': 'Unable to fix command configuration'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _apply_progressive_remediation(self, pod_id: str, pod: Any) -> Dict:
        """Apply progressive remediation strategies."""
        try:
            # Get remediation history
            history = self.state["remediation_history"].get(pod_id, [])
            attempt_count = len(history)
            
            # Progressive strategies based on attempt count
            if attempt_count == 0:
                # First attempt: Simple restart
                return self._restart_pod(pod_id)
            elif attempt_count == 1:
                # Second attempt: Adjust resource limits
                for container in pod.spec.containers:
                    if container.resources:
                        return self._adjust_resource_limits(pod_id, container)
            elif attempt_count == 2:
                # Third attempt: Check and fix configuration
                return self._validate_and_fix_config(pod_id, pod)
            elif attempt_count == 3:
                # Fourth attempt: Move pod to different node
                return self._reschedule_pod(pod_id, pod)
            else:
                # Final attempt: Scale down and up deployment
                return self._scale_deployment_zero_and_up(pod_id)
            
            return {'success': False, 'error': 'All remediation strategies exhausted'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _extract_error_patterns(self, logs: str) -> List[str]:
        """Extract error patterns from logs using regex and ML."""
        patterns = []
        try:
            # Use regex to find error patterns
            error_matches = re.finditer(
                r'(?i)(error|exception|failed|fatal|panic).*?[\n\.]',
                logs
            )
            patterns.extend(match.group(0) for match in error_matches)
            
            # Use ML model to identify error patterns
            if self.error_classifier:
                ml_patterns = self.error_classifier.extract_patterns(logs)
                patterns.extend(ml_patterns)
            
            return list(set(patterns))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
            return patterns

    def _create_rollback_point(self, pod_id: str):
        """Create a rollback point for a pod."""
        try:
            pod = self.tools.get_pod(pod_id)
            if pod:
                self.rollback_points[pod_id] = {
                    'timestamp': time.time(),
                    'pod_spec': pod.spec,
                    'pod_status': pod.status
                }
        except Exception as e:
            logger.error(f"Error creating rollback point: {str(e)}")

    def _rollback(self, pod_id: str) -> bool:
        """Rollback a pod to its previous state."""
        try:
            if pod_id in self.rollback_points:
                rollback_point = self.rollback_points[pod_id]
                pod = self.tools.get_pod(pod_id)
                if pod:
                    # Restore pod spec
                    pod.spec = rollback_point['pod_spec']
                    self.tools.update_pod(pod)
                    return True
            return False
        except Exception as e:
            logger.error(f"Error during rollback: {str(e)}")
            return False

    def _update_remediation_history(self, state: AgentState, pod_id: str, result: Dict):
        """Update remediation history with detailed information."""
        if pod_id not in state["remediation_history"]:
            state["remediation_history"][pod_id] = []
        
        state["remediation_history"][pod_id].append({
            "timestamp": time.time(),
            "action": result.get('action', 'unknown'),
            "success": result.get('success', False),
            "details": result.get('details', ''),
            "error": result.get('error', ''),
            "metrics_before": self.state_history[pod_id]['metrics'],
            "metrics_after": self.tools.get_pod_metrics(pod_id)
        })

    def _scale_deployment(self, deployment: Any, scale_factor: float) -> Dict:
        """Scale a deployment with safety checks."""
        try:
            current_replicas = deployment.spec.replicas or 1
            new_replicas = min(max(int(current_replicas * scale_factor), 1), 10)
            
            if new_replicas == current_replicas:
                return {'success': False, 'error': 'No scaling needed'}
            
            # Update deployment
            deployment.spec.replicas = new_replicas
            self.tools.update_deployment(deployment)
            
            return {
                'success': True,
                'action': 'scale_deployment',
                'details': f'Scaled from {current_replicas} to {new_replicas} replicas'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _update_hpa(self, hpa: Any, scale_factor: float) -> Dict:
        """Update HPA with new scaling parameters."""
        try:
            current_min = hpa.spec.min_replicas or 1
            new_min = min(max(int(current_min * scale_factor), 1), 10)
            
            if new_min == current_min:
                return {'success': False, 'error': 'No HPA update needed'}
            
            hpa.spec.min_replicas = new_min
            self.tools.update_hpa(hpa)
            
            return {
                'success': True,
                'action': 'update_hpa',
                'details': f'Updated HPA min replicas from {current_min} to {new_min}'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _restart_pod(self, pod_id: str) -> Dict:
        """Restart a pod with safety checks."""
        try:
            if self.tools.delete_pod(pod_id):
                return {
                    'success': True,
                    'action': 'restart_pod',
                    'details': 'Pod restarted successfully'
                }
            return {'success': False, 'error': 'Failed to restart pod'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _fix_dns_issues(self, pod_id: str) -> Dict:
        """Fix DNS-related issues."""
        try:
            pod = self.tools.get_pod(pod_id)
            if pod and pod.spec.dns_policy != 'ClusterFirst':
                pod.spec.dns_policy = 'ClusterFirst'
                self.tools.update_pod(pod)
                return {
                    'success': True,
                    'action': 'fix_dns',
                    'details': 'Updated DNS policy to ClusterFirst'
                }
            return {'success': False, 'error': 'No DNS policy update needed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _fix_network_policy(self, pod_id: str) -> Dict:
        """Fix network policy issues."""
        try:
            pod = self.tools.get_pod(pod_id)
            if pod:
                namespace = pod.metadata.namespace
                policies = self.tools.get_network_policies(namespace)
                for policy in policies:
                    if self._is_pod_affected_by_policy(pod, policy):
                        # Update policy to allow necessary traffic
                        self._update_network_policy(policy, pod)
                        return {
                            'success': True,
                            'action': 'update_network_policy',
                            'details': f'Updated network policy {policy.metadata.name}'
                        }
            return {'success': False, 'error': 'No network policy issues found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _fix_permissions(self, pod_id: str) -> Dict:
        """Fix permission-related issues."""
        try:
            pod = self.tools.get_pod(pod_id)
            if pod:
                # Check service account
                service_account = pod.spec.service_account_name
                if service_account:
                    # Verify and update RBAC if needed
                    return self._update_rbac(service_account, pod.metadata.namespace)
            return {'success': False, 'error': 'No permission issues found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _adjust_resource_limits(self, pod_id: str, container: Any) -> Dict:
        """Adjust resource limits for a container."""
        try:
            if container.resources.limits:
                # Increase limits by 20%
                new_limits = {}
                for resource, value in container.resources.limits.items():
                    new_limits[resource] = self._increase_resource_value(value, 1.2)
                container.resources.limits = new_limits
                
                # Update pod
                pod = self.tools.get_pod(pod_id)
                if pod:
                    self.tools.update_pod(pod)
                    return {
                        'success': True,
                        'action': 'adjust_resources',
                        'details': 'Increased resource limits by 20%'
                    }
            return {'success': False, 'error': 'No resource limits to adjust'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _increase_resource_value(self, value: str, factor: float) -> str:
        """Increase a resource value by a factor."""
        try:
            number = float(value[:-1])
            unit = value[-1]
            new_number = number * factor
            return f"{new_number}{unit}"
        except:
            return value

    def _is_pod_affected_by_policy(self, pod: Any, policy: Any) -> bool:
        """Check if a pod is affected by a network policy."""
        try:
            if not policy.spec.pod_selector:
                return True
            
            pod_labels = pod.metadata.labels or {}
            selector = policy.spec.pod_selector.match_labels or {}
            
            return all(pod_labels.get(k) == v for k, v in selector.items())
        except:
            return False

    def _update_network_policy(self, policy: Any, pod: Any):
        """Update a network policy to allow necessary traffic."""
        try:
            if not policy.spec.ingress:
                policy.spec.ingress = []
            
            # Add rule to allow all ingress traffic
            policy.spec.ingress.append({
                'from': [{}],
                'ports': [{'protocol': 'TCP'}]
            })
            
            self.tools.update_network_policy(policy)
        except Exception as e:
            logger.error(f"Error updating network policy: {str(e)}")

    def _update_rbac(self, service_account: str, namespace: str) -> Dict:
        """Update RBAC rules for a service account."""
        try:
            # Create or update role binding
            role_binding = self.tools.get_role_binding(service_account, namespace)
            if not role_binding:
                # Create new role binding
                self.tools.create_role_binding(service_account, namespace)
                return {
                    'success': True,
                    'action': 'create_rbac',
                    'details': f'Created role binding for {service_account}'
                }
            return {'success': False, 'error': 'Role binding already exists'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _is_node_pressured(self, node_metrics: Dict) -> bool:
        """Check if node is under resource pressure."""
        try:
            # Check CPU pressure
            cpu_pressure = node_metrics.get('cpu_pressure', False)
            if cpu_pressure:
                return True
            
            # Check memory pressure
            memory_pressure = node_metrics.get('memory_pressure', False)
            if memory_pressure:
                return True
            
            # Check disk pressure
            disk_pressure = node_metrics.get('disk_pressure', False)
            if disk_pressure:
                return True
            
            # Check PID pressure
            pid_pressure = node_metrics.get('pid_pressure', False)
            if pid_pressure:
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking node pressure: {str(e)}")
            return False

    def _handle_node_pressure(self, pod_id: str, node_name: str, node_metrics: Dict) -> Dict:
        """Handle node pressure issues."""
        try:
            # Get all pods on the node
            node_pods = self.tools.list_pods_on_node(node_name)
            
            # Calculate total resource usage
            total_cpu = 0
            total_memory = 0
            for pod in node_pods:
                pod_metrics = self.tools.get_pod_metrics(f"{pod.metadata.namespace}/{pod.metadata.name}")
                total_cpu += pod_metrics.get('cpu_usage', 0)
                total_memory += pod_metrics.get('memory_usage', 0)
            
            # If node is overloaded, try to move some pods
            if total_cpu > 80 or total_memory > 80:
                # Find another suitable node
                target_node = self._find_suitable_node(node_metrics)
                if target_node:
                    # Move the pod to the target node
                    return self._move_pod_to_node(pod_id, target_node)
                
            return {'success': False, 'error': 'Unable to handle node pressure'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _find_suitable_node(self, current_node_metrics: Dict) -> Optional[str]:
        """Find a suitable node for pod relocation."""
        try:
            nodes = self.tools.list_nodes()
            best_node = None
            lowest_usage = float('inf')
            
            for node in nodes:
                # Skip nodes that are not ready
                if not self._is_node_ready(node):
                    continue
                
                # Get node metrics
                metrics = self.tools.get_node_metrics(node.metadata.name)
                
                # Calculate node score based on resource usage
                cpu_usage = metrics.get('cpu_usage', 0)
                memory_usage = metrics.get('memory_usage', 0)
                score = (cpu_usage + memory_usage) / 2
                
                if score < lowest_usage:
                    lowest_usage = score
                    best_node = node.metadata.name
                
            return best_node
        except Exception as e:
            logger.error(f"Error finding suitable node: {str(e)}")
            return None

    def _move_pod_to_node(self, pod_id: str, target_node: str) -> Dict:
        """Move a pod to a target node."""
        try:
            pod = self.tools.get_pod(pod_id)
            if not pod:
                return {'success': False, 'error': 'Pod not found'}
            
            # Add node affinity to force pod to move
            pod.spec.affinity = {
                'nodeAffinity': {
                    'requiredDuringSchedulingIgnoredDuringExecution': {
                        'nodeSelectorTerms': [{
                            'matchExpressions': [{
                                'key': 'kubernetes.io/hostname',
                                'operator': 'In',
                                'values': [target_node]
                            }]
                        }]
                    }
                }
            }
            
            # Update the pod
            self.tools.update_pod(pod)
            
            return {
                'success': True,
                'action': 'move_pod',
                'details': f'Moving pod to node {target_node}'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _has_network_issues(self, logs: str) -> bool:
        """Check if logs indicate network issues."""
        network_patterns = [
            r'connection refused',
            r'network timeout',
            r'dns lookup failed',
            r'no route to host',
            r'connection reset',
            r'i/o timeout'
        ]
        
        return any(re.search(pattern, logs.lower()) for pattern in network_patterns)

    def _fix_network_issues(self, pod_id: str, pod: Any) -> Dict:
        """Fix network-related issues."""
        try:
            # Check DNS configuration
            if self._has_dns_issues(pod_id):
                return self._fix_dns_issues(pod_id)
            
            # Check network policies
            if self._has_network_policy_issues(pod):
                return self._fix_network_policy(pod_id)
            
            # Check service mesh configuration
            if self._has_service_mesh_issues(pod):
                return self._fix_service_mesh_config(pod_id)
            
            return {'success': False, 'error': 'Unable to determine network issue'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _scale_deployment_zero_and_up(self, pod_id: str) -> Dict:
        """Scale deployment to zero and back up."""
        try:
            # Get deployment for pod
            deployment = self._get_pod_deployment(pod_id)
            if not deployment:
                return {'success': False, 'error': 'No deployment found'}
            
            # Scale down to zero
            result = self._scale_deployment(deployment, 0)
            if not result['success']:
                return result
            
            time.sleep(10)  # Wait for scale down
            
            # Scale back up to original replicas
            original_replicas = deployment.spec.replicas or 1
            return self._scale_deployment(deployment, original_replicas)
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _validate_and_fix_config(self, pod_id: str, pod: Any) -> Dict:
        """Validate and fix pod configuration."""
        try:
            issues = []
            
            # Check container configuration
            for container in pod.spec.containers:
                # Validate resource requests and limits
                if not container.resources:
                    issues.append('Missing resource configuration')
                elif not container.resources.requests:
                    issues.append('Missing resource requests')
                
                # Validate probes
                if not container.readiness_probe:
                    issues.append('Missing readiness probe')
                if not container.liveness_probe:
                    issues.append('Missing liveness probe')
                
                # Validate security context
                if not container.security_context:
                    issues.append('Missing security context')
                
            if issues:
                # Fix the identified issues
                return self._fix_configuration_issues(pod_id, pod, issues)
            
            return {'success': False, 'error': 'No configuration issues found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

def create_agent_graph(llm: LlamaCpp):
    """Create the agent workflow graph."""
    
    if LANGGRAPH_AVAILABLE:
        # Create the graph using LangGraph
        workflow = StateGraph(AgentState)
        
        # Create agents
        monitoring_agent = MonitoringAgent(llm)
        analysis_agent = AnalysisAgent(llm)
        remediation_agent = RemediationAgent(llm)
        
        # Create Kubernetes tools
        k8s_state = KubernetesState(
            api_client=None,  # Will be set when running
            namespace="default",
            simulation_mode=True
        )
        k8s_tools = KubernetesTools(k8s_state)
        
        # Set up tools for agents
        monitoring_agent.setup_tools(k8s_tools)
        analysis_agent.setup_tools(k8s_tools)
        remediation_agent.setup_tools(k8s_tools)
        
        # Add nodes
        workflow.add_node("monitoring", monitoring_agent.process)
        workflow.add_node("analysis", analysis_agent.process)
        workflow.add_node("remediation", remediation_agent.process)
        
        # Add edges
        workflow.add_edge("monitoring", "analysis")
        workflow.add_conditional_edges(
            "analysis",
            lambda x: x["next_agent"],
            {
                "monitoring": "monitoring",
                "remediation": "remediation"
            }
        )
        workflow.add_edge("remediation", "monitoring")
        
        # Set entry point
        workflow.set_entry_point("monitoring")
        
        return workflow
    else:
        # Fallback to LangChain
        from langchain.agents import AgentExecutor, create_openapi_agent
        from langchain.agents.agent_types import AgentType
        from langchain.memory import ConversationBufferMemory
        
        # Create Kubernetes tools
        k8s_state = KubernetesState(
            api_client=None,  # Will be set when running
            namespace="default",
            simulation_mode=True
        )
        k8s_tools = KubernetesTools(k8s_state)
        
        # Create a simple agent executor
        tools = [
            k8s_tools.get_pod_metrics,
            k8s_tools.get_pod_logs,
            k8s_tools.restart_pod
        ]
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        agent = AgentExecutor.from_agent_and_tools(
            agent=create_openapi_agent(llm, tools, verbose=True),
            tools=tools,
            memory=memory,
            verbose=True
        )
        
        return agent

def run_mas():
    """Run the Multi-Agent System."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get model path from environment or use a default
        model_path = os.getenv("LLAMA_MODEL_PATH")
        if not model_path:
            logger.warning("LLAMA_MODEL_PATH not set, using default model path")
            # Use a default model path or download a model
            model_path = "models/llama-2-7b-chat.gguf"
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Check if the model exists
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}. Please set LLAMA_MODEL_PATH in .env file.")
                return
        
        # Create LLM
        llm = LlamaCpp(
            model_path=model_path,
            temperature=float(os.getenv("LLAMA_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "2000")),
            n_ctx=int(os.getenv("LLAMA_CONTEXT_SIZE", "4096"))
        )
        
        # Create the agent graph or executor
        workflow = create_agent_graph(llm)
        
        if LANGGRAPH_AVAILABLE:
            # Compile the graph
            app = workflow.compile()
            
            # Initialize state
            initial_state = AgentState(
                messages=[],
                next_agent="monitoring",
                current_agent="monitoring",
                metrics={},
                remediation_history={},
                last_remediation={},
                issues=[]
            )
            
            # Run the workflow
            logger.info("Starting Multi-Agent System with LangGraph")
            while True:
                try:
                    # Execute one step of the workflow
                    result = app.invoke(initial_state)
                    initial_state = result
                    
                    # Sleep for a bit before the next iteration
                    time.sleep(5)
                    
                except KeyboardInterrupt:
                    logger.info("Received shutdown signal")
                    break
                except Exception as e:
                    logger.error(f"Error in workflow execution: {str(e)}")
                    time.sleep(5)  # Wait before retrying
        else:
            # Run with LangChain
            logger.info("Starting Multi-Agent System with LangChain")
            while True:
                try:
                    # Execute the agent
                    result = workflow.invoke({"input": "Monitor Kubernetes pods and remediate issues"})
                    logger.info(f"Agent result: {result}")
                    
                    # Sleep for a bit before the next iteration
                    time.sleep(5)
                    
                except KeyboardInterrupt:
                    logger.info("Received shutdown signal")
                    break
                except Exception as e:
                    logger.error(f"Error in agent execution: {str(e)}")
                    time.sleep(5)  # Wait before retrying
        
        logger.info("Multi-Agent System shutdown complete")
        
    except Exception as e:
        logger.error(f"Error in Multi-Agent System: {str(e)}")
        raise

if __name__ == "__main__":
    run_mas() 