from typing import Dict, Any, List, Optional
from langchain_core.tools import Tool
from .langchain_base_agent import LangChainBaseAgent
from kubernetes import client
import json

class ResourceExhaustionAgent(LangChainBaseAgent):
    def __init__(self, k8s_api: client.CoreV1Api, k8s_apps_api: client.AppsV1Api):
        super().__init__("resource-exhaustion", k8s_api, k8s_apps_api)
        self.cpu_threshold = 80.0
        self.memory_threshold = 80.0
        self.max_scale_factor = 2.0

    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="scale_deployment",
                func=self._scale_deployment,
                description="Scale a Kubernetes deployment to a new number of replicas"
            ),
            Tool(
                name="get_deployment_info",
                func=self._get_deployment_info,
                description="Get information about a Kubernetes deployment"
            ),
            Tool(
                name="analyze_resource_usage",
                func=self._analyze_resource_usage,
                description="Analyze resource usage patterns and determine if scaling is needed"
            )
        ]

    def _scale_deployment(self, deployment_name: str, namespace: str, new_replicas: int) -> str:
        """Scale a deployment to a new number of replicas"""
        try:
            deployment = self.k8s_apps_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            current_replicas = deployment.spec.replicas or 1
            new_replicas = min(max(new_replicas, 1), 10)  # Ensure between 1 and 10
            
            if new_replicas != current_replicas:
                deployment.spec.replicas = new_replicas
                self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment
                )
                return f"Successfully scaled deployment {deployment_name} from {current_replicas} to {new_replicas} replicas"
            return f"Deployment {deployment_name} already at {current_replicas} replicas"
            
        except Exception as e:
            return f"Error scaling deployment: {str(e)}"

    def _get_deployment_info(self, deployment_name: str, namespace: str) -> str:
        """Get information about a deployment"""
        try:
            deployment = self.k8s_apps_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            return json.dumps({
                'name': deployment.metadata.name,
                'replicas': deployment.spec.replicas,
                'available_replicas': deployment.status.available_replicas,
                'ready_replicas': deployment.status.ready_replicas,
                'updated_replicas': deployment.status.updated_replicas
            }, indent=2)
            
        except Exception as e:
            return f"Error getting deployment info: {str(e)}"

    def _analyze_resource_usage(self, metrics: Dict[str, Any]) -> str:
        """Analyze resource usage and provide recommendations"""
        cpu_usage = float(metrics.get('CPU Usage (%)', 0.0))
        memory_usage = float(metrics.get('Memory Usage (%)', 0.0))
        
        analysis = {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'cpu_threshold_exceeded': cpu_usage > self.cpu_threshold,
            'memory_threshold_exceeded': memory_usage > self.memory_threshold,
            'recommendations': []
        }
        
        if cpu_usage > self.cpu_threshold:
            severity = (cpu_usage - self.cpu_threshold) / 100
            scale_factor = min(1.0 + severity, self.max_scale_factor)
            analysis['recommendations'].append({
                'action': 'scale_deployment',
                'reason': 'High CPU usage',
                'severity': severity,
                'scale_factor': scale_factor
            })
            
        if memory_usage > self.memory_threshold:
            severity = (memory_usage - self.memory_threshold) / 100
            scale_factor = min(1.0 + severity, self.max_scale_factor)
            analysis['recommendations'].append({
                'action': 'scale_deployment',
                'reason': 'High memory usage',
                'severity': severity,
                'scale_factor': scale_factor
            })
            
        return json.dumps(analysis, indent=2)

    def _parse_agent_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the agent's result into a structured format"""
        try:
            # Extract the action from the agent's response
            response = result.get('output', '')
            
            if 'Successfully scaled deployment' in response:
                return {
                    'action_taken': True,
                    'action': 'scale_deployment',
                    'details': response
                }
            elif 'Error' in response:
                return {
                    'action_taken': False,
                    'error': response
                }
            else:
                return {
                    'action_taken': False,
                    'reason': 'No action needed',
                    'details': response
                }
                
        except Exception as e:
            return {
                'action_taken': False,
                'error': f"Error parsing agent result: {str(e)}"
            }

class CrashLoopAgent(LangChainBaseAgent):
    def __init__(self, k8s_api: client.CoreV1Api, k8s_apps_api: client.AppsV1Api):
        super().__init__("crash-loop", k8s_api, k8s_apps_api)
        self.restart_threshold = 5
        self.max_restarts = 10

    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="get_pod_logs",
                func=self._get_pod_logs,
                description="Get logs from a Kubernetes pod"
            ),
            Tool(
                name="delete_pod",
                func=self._delete_pod,
                description="Delete a Kubernetes pod to force a restart"
            ),
            Tool(
                name="analyze_crash_pattern",
                func=self._analyze_crash_pattern,
                description="Analyze pod crash patterns and determine the root cause"
            )
        ]

    def _get_pod_logs(self, pod_name: str, namespace: str) -> str:
        """Get logs from a pod"""
        try:
            logs = self.k8s_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace
            )
            return logs
        except Exception as e:
            return f"Error getting pod logs: {str(e)}"

    def _delete_pod(self, pod_name: str, namespace: str) -> str:
        """Delete a pod to force a restart"""
        try:
            self.k8s_api.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace
            )
            return f"Successfully deleted pod {pod_name}"
        except Exception as e:
            return f"Error deleting pod: {str(e)}"

    def _analyze_crash_pattern(self, metrics: Dict[str, Any], logs: str) -> str:
        """Analyze crash patterns and determine root cause"""
        restarts = int(metrics.get('Pod Restarts', 0))
        
        analysis = {
            'restarts': restarts,
            'severity': min(restarts / self.max_restarts, 1.0),
            'needs_action': restarts >= self.restart_threshold,
            'should_delete': restarts >= self.max_restarts,
            'log_analysis': self._analyze_logs(logs)
        }
        
        return json.dumps(analysis, indent=2)

    def _analyze_logs(self, logs: str) -> Dict[str, Any]:
        """Analyze pod logs for error patterns"""
        # This is a placeholder - in a real implementation, you would use
        # more sophisticated log analysis techniques
        return {
            'error_count': logs.count('ERROR'),
            'warning_count': logs.count('WARNING'),
            'last_error': logs.split('\n')[-1] if logs else None
        }

    def _parse_agent_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the agent's result into a structured format"""
        try:
            response = result.get('output', '')
            
            if 'Successfully deleted pod' in response:
                return {
                    'action_taken': True,
                    'action': 'delete_pod',
                    'details': response
                }
            elif 'Error' in response:
                return {
                    'action_taken': False,
                    'error': response
                }
            else:
                return {
                    'action_taken': False,
                    'reason': 'No action needed',
                    'details': response
                }
                
        except Exception as e:
            return {
                'action_taken': False,
                'error': f"Error parsing agent result: {str(e)}"
            }

class NetworkIssueAgent(LangChainBaseAgent):
    def __init__(self, k8s_api: client.CoreV1Api, k8s_apps_api: client.AppsV1Api):
        super().__init__("network-issue", k8s_api, k8s_apps_api)
        self.packet_loss_threshold = 100  # packets per second
        self.retry_threshold = 3

    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="restart_pod",
                func=self._restart_pod,
                description="Restart a pod to resolve network issues"
            ),
            Tool(
                name="analyze_network_metrics",
                func=self._analyze_network_metrics,
                description="Analyze network metrics and determine if action is needed"
            ),
            Tool(
                name="check_network_policy",
                func=self._check_network_policy,
                description="Check network policies affecting a pod"
            )
        ]

    def _restart_pod(self, pod_name: str, namespace: str) -> str:
        """Restart a pod to resolve network issues"""
        try:
            self.k8s_api.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace
            )
            return f"Successfully restarted pod {pod_name}"
        except Exception as e:
            return f"Error restarting pod: {str(e)}"

    def _analyze_network_metrics(self, metrics: Dict[str, Any]) -> str:
        """Analyze network metrics and provide recommendations"""
        rx_dropped = float(metrics.get('Network Receive Packets Dropped (p/s)', 0.0))
        tx_dropped = float(metrics.get('Network Transmit Packets Dropped (p/s)', 0.0))
        
        analysis = {
            'rx_dropped': rx_dropped,
            'tx_dropped': tx_dropped,
            'rx_threshold_exceeded': rx_dropped > self.packet_loss_threshold,
            'tx_threshold_exceeded': tx_dropped > self.packet_loss_threshold,
            'needs_action': rx_dropped > self.packet_loss_threshold or tx_dropped > self.packet_loss_threshold,
            'severity': max(
                rx_dropped / self.packet_loss_threshold if rx_dropped > self.packet_loss_threshold else 0,
                tx_dropped / self.packet_loss_threshold if tx_dropped > self.packet_loss_threshold else 0
            )
        }
        
        return json.dumps(analysis, indent=2)

    def _check_network_policy(self, pod_name: str, namespace: str) -> str:
        """Check network policies affecting a pod"""
        try:
            # Get network policies in the namespace
            policies = self.k8s_api.list_namespaced_network_policy(namespace=namespace)
            
            # Get pod labels
            pod = self.k8s_api.read_namespaced_pod(name=pod_name, namespace=namespace)
            pod_labels = pod.metadata.labels or {}
            
            # Check which policies affect this pod
            affecting_policies = []
            for policy in policies.items:
                if self._policy_affects_pod(policy, pod_labels):
                    affecting_policies.append({
                        'name': policy.metadata.name,
                        'namespace': policy.metadata.namespace
                    })
            
            return json.dumps({
                'pod_name': pod_name,
                'namespace': namespace,
                'affecting_policies': affecting_policies
            }, indent=2)
            
        except Exception as e:
            return f"Error checking network policies: {str(e)}"

    def _policy_affects_pod(self, policy, pod_labels: Dict[str, str]) -> bool:
        """Check if a network policy affects a pod based on labels"""
        # This is a simplified check - in a real implementation, you would
        # need to check all policy rules and match expressions
        if not policy.spec.pod_selector:
            return True
            
        selector = policy.spec.pod_selector.match_labels or {}
        return all(pod_labels.get(k) == v for k, v in selector.items())

    def _parse_agent_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the agent's result into a structured format"""
        try:
            response = result.get('output', '')
            
            if 'Successfully restarted pod' in response:
                return {
                    'action_taken': True,
                    'action': 'restart_pod',
                    'details': response
                }
            elif 'Error' in response:
                return {
                    'action_taken': False,
                    'error': response
                }
            else:
                return {
                    'action_taken': False,
                    'reason': 'No action needed',
                    'details': response
                }
                
        except Exception as e:
            return {
                'action_taken': False,
                'error': f"Error parsing agent result: {str(e)}"
            } 