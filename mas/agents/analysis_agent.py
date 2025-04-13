from .base_agent import BaseAgent
from langchain.agents import Tool
from typing import List, Dict, Any
import kubernetes as k8s
import os
import json
from datetime import datetime, timedelta

class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing issues in the Kubernetes cluster"""
    
    def __init__(self, llm):
        """Initialize the analysis agent"""
        super().__init__(llm)
        self.k8s_client = k8s.client.CoreV1Api()
        self.apps_client = k8s.client.AppsV1Api()
        self.events_client = k8s.client.EventsV1Api()
    
    def get_tools(self) -> List[Tool]:
        """Get the analysis tools available to this agent"""
        return [
            Tool(
                name="analyze_pod_logs",
                func=self._analyze_pod_logs,
                description="Analyze logs from a pod to identify issues"
            ),
            Tool(
                name="get_pod_events",
                func=self._get_pod_events,
                description="Get events related to a pod"
            ),
            Tool(
                name="analyze_resource_usage",
                func=self._analyze_resource_usage,
                description="Analyze resource usage patterns for a pod or deployment"
            ),
            Tool(
                name="detect_anomalies",
                func=self._detect_anomalies,
                description="Detect anomalies in pod or deployment behavior"
            )
        ]
    
    def _analyze_pod_logs(self, pod_name: str, namespace: str = "default", 
                         container: str = None, lines: int = 100) -> Dict[str, Any]:
        """Analyze logs from a pod to identify issues"""
        try:
            # Get pod logs
            logs = self.k8s_client.read_namespaced_pod_log(
                pod_name, 
                namespace,
                container=container,
                tail_lines=lines
            )
            
            # Use LLM to analyze logs
            analysis_prompt = f"""
            Analyze the following Kubernetes pod logs and identify any issues or anomalies:
            
            {logs}
            
            Provide a structured analysis with:
            1. Identified issues
            2. Severity level (low, medium, high)
            3. Recommended actions
            """
            
            analysis_result = self.llm.invoke(analysis_prompt)
            
            return {
                'pod_name': pod_name,
                'namespace': namespace,
                'container': container,
                'analysis': analysis_result,
                'log_sample': logs[:500] + "..." if len(logs) > 500 else logs
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _get_pod_events(self, pod_name: str, namespace: str = "default", 
                       hours: int = 24) -> Dict[str, Any]:
        """Get events related to a pod"""
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Get events
            events = self.events_client.list_namespaced_event(
                namespace,
                field_selector=f"involvedObject.name={pod_name},involvedObject.kind=Pod"
            )
            
            # Filter events by time
            recent_events = [
                event for event in events.items
                if event.last_timestamp and event.last_timestamp > start_time
            ]
            
            # Format events
            formatted_events = [
                {
                    'type': event.type,
                    'reason': event.reason,
                    'message': event.message,
                    'time': event.last_timestamp.isoformat() if event.last_timestamp else None,
                    'count': event.count
                }
                for event in recent_events
            ]
            
            return {
                'pod_name': pod_name,
                'namespace': namespace,
                'events': formatted_events,
                'count': len(formatted_events)
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _analyze_resource_usage(self, target: str, namespace: str = "default", 
                               resource_type: str = "pod") -> Dict[str, Any]:
        """Analyze resource usage patterns for a pod or deployment"""
        try:
            if resource_type.lower() == "pod":
                # Get pod metrics
                pod = self.k8s_client.read_namespaced_pod(target, namespace)
                containers = pod.spec.containers
                
                # Analyze resource requests and limits
                resource_analysis = []
                for container in containers:
                    requests = container.resources.requests if container.resources and container.resources.requests else {}
                    limits = container.resources.limits if container.resources and container.resources.limits else {}
                    
                    resource_analysis.append({
                        'container_name': container.name,
                        'cpu_request': requests.get('cpu', 'Not set'),
                        'memory_request': requests.get('memory', 'Not set'),
                        'cpu_limit': limits.get('cpu', 'Not set'),
                        'memory_limit': limits.get('memory', 'Not set')
                    })
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'pod',
                    'resource_analysis': resource_analysis
                }
            elif resource_type.lower() == "deployment":
                # Get deployment
                deployment = self.apps_client.read_namespaced_deployment(target, namespace)
                containers = deployment.spec.template.spec.containers
                
                # Analyze resource requests and limits
                resource_analysis = []
                for container in containers:
                    requests = container.resources.requests if container.resources and container.resources.requests else {}
                    limits = container.resources.limits if container.resources and container.resources.limits else {}
                    
                    resource_analysis.append({
                        'container_name': container.name,
                        'cpu_request': requests.get('cpu', 'Not set'),
                        'memory_request': requests.get('memory', 'Not set'),
                        'cpu_limit': limits.get('cpu', 'Not set'),
                        'memory_limit': limits.get('memory', 'Not set')
                    })
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'deployment',
                    'resource_analysis': resource_analysis
                }
            else:
                return {
                    'error': f"Unsupported resource type: {resource_type}"
                }
        except Exception as e:
            return self.handle_error(e)
    
    def _detect_anomalies(self, target: str, namespace: str = "default", 
                         resource_type: str = "pod") -> Dict[str, Any]:
        """Detect anomalies in pod or deployment behavior"""
        try:
            if resource_type.lower() == "pod":
                # Get pod status
                pod = self.k8s_client.read_namespaced_pod(target, namespace)
                
                # Check for common anomalies
                anomalies = []
                
                # Check pod phase
                if pod.status.phase not in ["Running", "Succeeded"]:
                    anomalies.append({
                        'type': 'pod_phase',
                        'severity': 'high',
                        'description': f"Pod is in {pod.status.phase} phase"
                    })
                
                # Check container statuses
                if pod.status.container_statuses:
                    for container in pod.status.container_statuses:
                        # Check restart count
                        if container.restart_count > 5:
                            anomalies.append({
                                'type': 'high_restart_count',
                                'severity': 'high',
                                'description': f"Container {container.name} has restarted {container.restart_count} times"
                            })
                        
                        # Check ready status
                        if not container.ready:
                            anomalies.append({
                                'type': 'container_not_ready',
                                'severity': 'medium',
                                'description': f"Container {container.name} is not ready"
                            })
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'pod',
                    'anomalies': anomalies,
                    'count': len(anomalies)
                }
            elif resource_type.lower() == "deployment":
                # Get deployment status
                deployment = self.apps_client.read_namespaced_deployment(target, namespace)
                
                # Check for common anomalies
                anomalies = []
                
                # Check available replicas
                if deployment.status.available_replicas != deployment.spec.replicas:
                    anomalies.append({
                        'type': 'replica_mismatch',
                        'severity': 'medium',
                        'description': f"Available replicas ({deployment.status.available_replicas}) don't match desired replicas ({deployment.spec.replicas})"
                    })
                
                # Check for failed deployments
                if deployment.status.conditions:
                    for condition in deployment.status.conditions:
                        if condition.type == "Available" and condition.status != "True":
                            anomalies.append({
                                'type': 'deployment_not_available',
                                'severity': 'high',
                                'description': f"Deployment is not available: {condition.message}"
                            })
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'deployment',
                    'anomalies': anomalies,
                    'count': len(anomalies)
                }
            else:
                return {
                    'error': f"Unsupported resource type: {resource_type}"
                }
        except Exception as e:
            return self.handle_error(e)
    
    def process_result(self, result: str) -> Dict[str, Any]:
        """Process the result from analysis actions"""
        # TODO: Implement result processing logic
        return {
            'status': 'processed',
            'result': result
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for analysis"""
        required_fields = ['action', 'target']
        return all(field in input_data for field in required_fields)
    
    def format_output(self, output_data: Dict[str, Any]) -> str:
        """Format the analysis output data"""
        # TODO: Implement output formatting logic
        return str(output_data) 