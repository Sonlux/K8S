from .base_agent import BaseAgent
from langchain.agents import Tool
from typing import List, Dict, Any
import kubernetes as k8s
import os
from ..utils.prometheus_client import PrometheusClient

class ResourceAgent(BaseAgent):
    """Agent responsible for managing resources in the Kubernetes cluster"""
    
    def __init__(self, llm):
        """Initialize the resource agent"""
        super().__init__(llm)
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        self.prometheus = PrometheusClient(base_url=prometheus_url)
        self.k8s_client = k8s.client.CoreV1Api()
        self.apps_client = k8s.client.AppsV1Api()
    
    def get_tools(self) -> List[Tool]:
        """Get the resource management tools available to this agent"""
        return [
            Tool(
                name="get_resource_usage",
                func=self._get_resource_usage,
                description="Get resource usage for a pod, node, or deployment"
            ),
            Tool(
                name="optimize_resources",
                func=self._optimize_resources,
                description="Optimize resource requests and limits based on usage patterns"
            ),
            Tool(
                name="scale_resources",
                func=self._scale_resources,
                description="Scale resources for a deployment based on demand"
            ),
            Tool(
                name="analyze_resource_trends",
                func=self._analyze_resource_trends,
                description="Analyze resource usage trends over time"
            )
        ]
    
    def _get_resource_usage(self, target: str, namespace: str = "default", 
                           resource_type: str = "pod") -> Dict[str, Any]:
        """Get resource usage for a pod, node, or deployment"""
        try:
            if resource_type.lower() == "pod":
                # Get pod metrics from Prometheus
                cpu_result = self.prometheus.query(
                    f'container_cpu_usage_seconds_total{{pod="{target}",namespace="{namespace}"}}'
                )
                memory_result = self.prometheus.query(
                    f'container_memory_usage_bytes{{pod="{target}",namespace="{namespace}"}}'
                )
                
                # Extract values from Prometheus response
                cpu_usage = cpu_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if cpu_result else 0
                memory_usage = memory_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if memory_result else 0
                
                # Get pod resource requests and limits
                pod = self.k8s_client.read_namespaced_pod(target, namespace)
                containers = pod.spec.containers
                
                resource_specs = []
                for container in containers:
                    requests = container.resources.requests if container.resources and container.resources.requests else {}
                    limits = container.resources.limits if container.resources and container.resources.limits else {}
                    
                    resource_specs.append({
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
                    'cpu_usage': float(cpu_usage),
                    'memory_usage': float(memory_usage),
                    'resource_specs': resource_specs
                }
            elif resource_type.lower() == "node":
                # Get node metrics from Prometheus
                cpu_result = self.prometheus.query(
                    f'node_cpu_seconds_total{{node="{target}"}}'
                )
                memory_result = self.prometheus.query(
                    f'node_memory_MemTotal_bytes{{node="{target}"}}'
                )
                
                # Extract values from Prometheus response
                cpu_usage = cpu_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if cpu_result else 0
                memory_total = memory_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if memory_result else 0
                
                return {
                    'target': target,
                    'resource_type': 'node',
                    'cpu_usage': float(cpu_usage),
                    'memory_total': float(memory_total)
                }
            elif resource_type.lower() == "deployment":
                # Get deployment metrics
                deployment = self.apps_client.read_namespaced_deployment(target, namespace)
                containers = deployment.spec.template.spec.containers
                
                # Get resource requests and limits
                resource_specs = []
                for container in containers:
                    requests = container.resources.requests if container.resources and container.resources.requests else {}
                    limits = container.resources.limits if container.resources and container.resources.limits else {}
                    
                    resource_specs.append({
                        'container_name': container.name,
                        'cpu_request': requests.get('cpu', 'Not set'),
                        'memory_request': requests.get('memory', 'Not set'),
                        'cpu_limit': limits.get('cpu', 'Not set'),
                        'memory_limit': limits.get('memory', 'Not set')
                    })
                
                # Get pod metrics for the deployment
                pod_selector = deployment.spec.selector.match_labels
                selector_string = ",".join([f"{k}={v}" for k, v in pod_selector.items()])
                
                cpu_result = self.prometheus.query(
                    f'sum(container_cpu_usage_seconds_total{{namespace="{namespace}",{selector_string}}})'
                )
                memory_result = self.prometheus.query(
                    f'sum(container_memory_usage_bytes{{namespace="{namespace}",{selector_string}}})'
                )
                
                # Extract values from Prometheus response
                cpu_usage = cpu_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if cpu_result else 0
                memory_usage = memory_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if memory_result else 0
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'deployment',
                    'cpu_usage': float(cpu_usage),
                    'memory_usage': float(memory_usage),
                    'resource_specs': resource_specs,
                    'replicas': deployment.spec.replicas
                }
            else:
                return {
                    'error': f"Unsupported resource type: {resource_type}"
                }
        except Exception as e:
            return self.handle_error(e)
    
    def _optimize_resources(self, target: str, namespace: str = "default", 
                           resource_type: str = "deployment") -> Dict[str, Any]:
        """Optimize resource requests and limits based on usage patterns"""
        try:
            if resource_type.lower() != "deployment":
                return {
                    'error': f"Resource optimization is only supported for deployments"
                }
            
            # Get current resource usage
            usage = self._get_resource_usage(target, namespace, resource_type)
            
            # Get deployment
            deployment = self.apps_client.read_namespaced_deployment(target, namespace)
            
            # Calculate optimal resources based on usage
            # This is a simplified example - in a real system, you would use more sophisticated algorithms
            cpu_usage = usage.get('cpu_usage', 0)
            memory_usage = usage.get('memory_usage', 0)
            
            # Add 20% buffer for CPU and 30% for memory
            optimal_cpu = f"{cpu_usage * 1.2:.2f}"
            optimal_memory = f"{memory_usage * 1.3:.0f}"
            
            # Update resource limits
            for container in deployment.spec.template.spec.containers:
                if not container.resources:
                    container.resources = k8s.client.V1ResourceRequirements()
                if not container.resources.limits:
                    container.resources.limits = {}
                if not container.resources.requests:
                    container.resources.requests = {}
                
                # Set limits
                container.resources.limits['cpu'] = optimal_cpu
                container.resources.limits['memory'] = optimal_memory
                
                # Set requests to 70% of limits
                container.resources.requests['cpu'] = f"{float(optimal_cpu) * 0.7:.2f}"
                container.resources.requests['memory'] = f"{float(optimal_memory) * 0.7:.0f}"
            
            # Apply the update
            self.apps_client.patch_namespaced_deployment(target, namespace, deployment)
            
            return {
                'target': target,
                'namespace': namespace,
                'resource_type': 'deployment',
                'optimized_cpu_limit': optimal_cpu,
                'optimized_memory_limit': optimal_memory,
                'message': f"Resource limits optimized for deployment {target} in namespace {namespace}"
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _scale_resources(self, target: str, namespace: str = "default", 
                        cpu_factor: float = 1.5, memory_factor: float = 1.5) -> Dict[str, Any]:
        """Scale resources for a deployment based on demand"""
        try:
            # Get deployment
            deployment = self.apps_client.read_namespaced_deployment(target, namespace)
            
            # Scale resources for all containers
            for container in deployment.spec.template.spec.containers:
                if not container.resources:
                    container.resources = k8s.client.V1ResourceRequirements()
                if not container.resources.limits:
                    container.resources.limits = {}
                if not container.resources.requests:
                    container.resources.requests = {}
                
                # Get current limits
                current_cpu_limit = container.resources.limits.get('cpu', '100m')
                current_memory_limit = container.resources.limits.get('memory', '128Mi')
                
                # Parse current limits
                cpu_value = float(current_cpu_limit[:-1]) if current_cpu_limit.endswith('m') else float(current_cpu_limit)
                memory_value = float(current_memory_limit[:-2]) if current_memory_limit.endswith('Mi') else float(current_memory_limit)
                
                # Calculate new limits
                new_cpu_limit = f"{cpu_value * cpu_factor:.0f}m"
                new_memory_limit = f"{memory_value * memory_factor:.0f}Mi"
                
                # Update limits
                container.resources.limits['cpu'] = new_cpu_limit
                container.resources.limits['memory'] = new_memory_limit
                
                # Update requests to 70% of limits
                container.resources.requests['cpu'] = f"{cpu_value * cpu_factor * 0.7:.0f}m"
                container.resources.requests['memory'] = f"{memory_value * memory_factor * 0.7:.0f}Mi"
            
            # Apply the update
            self.apps_client.patch_namespaced_deployment(target, namespace, deployment)
            
            return {
                'target': target,
                'namespace': namespace,
                'cpu_factor': cpu_factor,
                'memory_factor': memory_factor,
                'message': f"Resources scaled for deployment {target} in namespace {namespace}"
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _analyze_resource_trends(self, target: str, namespace: str = "default", 
                               resource_type: str = "pod", hours: int = 24) -> Dict[str, Any]:
        """Analyze resource usage trends over time"""
        try:
            if resource_type.lower() == "pod":
                # Get pod metrics from Prometheus for the last 24 hours
                cpu_result = self.prometheus.query_range(
                    f'container_cpu_usage_seconds_total{{pod="{target}",namespace="{namespace}"}}',
                    start=int((datetime.now() - timedelta(hours=hours)).timestamp()),
                    end=int(datetime.now().timestamp()),
                    step="5m"
                )
                memory_result = self.prometheus.query_range(
                    f'container_memory_usage_bytes{{pod="{target}",namespace="{namespace}"}}',
                    start=int((datetime.now() - timedelta(hours=hours)).timestamp()),
                    end=int(datetime.now().timestamp()),
                    step="5m"
                )
                
                # Extract values from Prometheus response
                cpu_trends = []
                memory_trends = []
                
                if cpu_result and 'data' in cpu_result and 'result' in cpu_result['data']:
                    for result in cpu_result['data']['result']:
                        for value in result.get('values', []):
                            cpu_trends.append({
                                'timestamp': value[0],
                                'value': float(value[1])
                            })
                
                if memory_result and 'data' in memory_result and 'result' in memory_result['data']:
                    for result in memory_result['data']['result']:
                        for value in result.get('values', []):
                            memory_trends.append({
                                'timestamp': value[0],
                                'value': float(value[1])
                            })
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'pod',
                    'cpu_trends': cpu_trends,
                    'memory_trends': memory_trends,
                    'hours': hours
                }
            elif resource_type.lower() == "deployment":
                # Get deployment pod selector
                deployment = self.apps_client.read_namespaced_deployment(target, namespace)
                pod_selector = deployment.spec.selector.match_labels
                selector_string = ",".join([f"{k}={v}" for k, v in pod_selector.items()])
                
                # Get deployment metrics from Prometheus for the last 24 hours
                cpu_result = self.prometheus.query_range(
                    f'sum(container_cpu_usage_seconds_total{{namespace="{namespace}",{selector_string}}})',
                    start=int((datetime.now() - timedelta(hours=hours)).timestamp()),
                    end=int(datetime.now().timestamp()),
                    step="5m"
                )
                memory_result = self.prometheus.query_range(
                    f'sum(container_memory_usage_bytes{{namespace="{namespace}",{selector_string}}})',
                    start=int((datetime.now() - timedelta(hours=hours)).timestamp()),
                    end=int(datetime.now().timestamp()),
                    step="5m"
                )
                
                # Extract values from Prometheus response
                cpu_trends = []
                memory_trends = []
                
                if cpu_result and 'data' in cpu_result and 'result' in cpu_result['data']:
                    for result in cpu_result['data']['result']:
                        for value in result.get('values', []):
                            cpu_trends.append({
                                'timestamp': value[0],
                                'value': float(value[1])
                            })
                
                if memory_result and 'data' in memory_result and 'result' in memory_result['data']:
                    for result in memory_result['data']['result']:
                        for value in result.get('values', []):
                            memory_trends.append({
                                'timestamp': value[0],
                                'value': float(value[1])
                            })
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'deployment',
                    'cpu_trends': cpu_trends,
                    'memory_trends': memory_trends,
                    'hours': hours
                }
            else:
                return {
                    'error': f"Unsupported resource type: {resource_type}"
                }
        except Exception as e:
            return self.handle_error(e)
    
    def process_result(self, result: str) -> Dict[str, Any]:
        """Process the result from resource management actions"""
        # TODO: Implement result processing logic
        return {
            'status': 'processed',
            'result': result
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for resource management"""
        required_fields = ['action', 'target']
        return all(field in input_data for field in required_fields)
    
    def format_output(self, output_data: Dict[str, Any]) -> str:
        """Format the resource management output data"""
        # TODO: Implement output formatting logic
        return str(output_data) 