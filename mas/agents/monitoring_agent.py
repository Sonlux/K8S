from .base_agent import BaseAgent
from langchain.agents import Tool
from typing import List, Dict, Any
import kubernetes as k8s
import os
from ..utils.prometheus_client import PrometheusClient

class MonitoringAgent(BaseAgent):
    """Agent responsible for monitoring Kubernetes cluster metrics and status"""
    
    def __init__(self, llm):
        """Initialize the monitoring agent"""
        super().__init__(llm)
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        self.prometheus = PrometheusClient(base_url=prometheus_url)
        self.k8s_client = k8s.client.CoreV1Api()
    
    def get_tools(self) -> List[Tool]:
        """Get the monitoring tools available to this agent"""
        return [
            Tool(
                name="get_pod_metrics",
                func=self._get_pod_metrics,
                description="Get metrics for a specific pod"
            ),
            Tool(
                name="get_node_metrics",
                func=self._get_node_metrics,
                description="Get metrics for a specific node"
            ),
            Tool(
                name="get_cluster_metrics",
                func=self._get_cluster_metrics,
                description="Get overall cluster metrics"
            ),
            Tool(
                name="check_pod_status",
                func=self._check_pod_status,
                description="Check the status of a specific pod"
            )
        ]
    
    def _get_pod_metrics(self, pod_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get metrics for a specific pod"""
        try:
            # Get pod metrics from Prometheus
            cpu_result = self.prometheus.query(
                f'container_cpu_usage_seconds_total{{pod="{pod_name}",namespace="{namespace}"}}'
            )
            memory_result = self.prometheus.query(
                f'container_memory_usage_bytes{{pod="{pod_name}",namespace="{namespace}"}}'
            )
            
            # Extract values from Prometheus response
            cpu_usage = cpu_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if cpu_result else 0
            memory_usage = memory_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if memory_result else 0
            
            return {
                'pod_name': pod_name,
                'namespace': namespace,
                'cpu_usage': float(cpu_usage),
                'memory_usage': float(memory_usage)
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _get_node_metrics(self, node_name: str) -> Dict[str, Any]:
        """Get metrics for a specific node"""
        try:
            # Get node metrics from Prometheus
            cpu_result = self.prometheus.query(
                f'node_cpu_seconds_total{{node="{node_name}"}}'
            )
            memory_result = self.prometheus.query(
                f'node_memory_MemTotal_bytes{{node="{node_name}"}}'
            )
            
            # Extract values from Prometheus response
            cpu_usage = cpu_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if cpu_result else 0
            memory_total = memory_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if memory_result else 0
            
            return {
                'node_name': node_name,
                'cpu_usage': float(cpu_usage),
                'memory_total': float(memory_total)
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _get_cluster_metrics(self) -> Dict[str, Any]:
        """Get overall cluster metrics"""
        try:
            # Get cluster-wide metrics from Prometheus
            nodes_result = self.prometheus.query('count(kube_node_info)')
            pods_result = self.prometheus.query('count(kube_pod_info)')
            cpu_result = self.prometheus.query('sum(node_cpu_seconds_total)')
            memory_result = self.prometheus.query('sum(node_memory_MemTotal_bytes)')
            
            # Extract values from Prometheus response
            total_nodes = nodes_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if nodes_result else 0
            total_pods = pods_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if pods_result else 0
            cluster_cpu = cpu_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if cpu_result else 0
            cluster_memory = memory_result.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if memory_result else 0
            
            return {
                'total_nodes': int(total_nodes),
                'total_pods': int(total_pods),
                'cluster_cpu': float(cluster_cpu),
                'cluster_memory': float(cluster_memory)
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _check_pod_status(self, pod_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Check the status of a specific pod"""
        try:
            pod = self.k8s_client.read_namespaced_pod(pod_name, namespace)
            return {
                'pod_name': pod_name,
                'namespace': namespace,
                'status': pod.status.phase,
                'containers': [
                    {
                        'name': container.name,
                        'ready': container.ready,
                        'restart_count': container.restart_count
                    }
                    for container in pod.status.container_statuses or []
                ]
            }
        except Exception as e:
            return self.handle_error(e)
    
    def process_result(self, result: str) -> Dict[str, Any]:
        """Process the result from monitoring actions"""
        # TODO: Implement result processing logic
        return {
            'status': 'processed',
            'result': result
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for monitoring"""
        required_fields = ['action', 'target']
        return all(field in input_data for field in required_fields)
    
    def format_output(self, output_data: Dict[str, Any]) -> str:
        """Format the monitoring output data"""
        # TODO: Implement output formatting logic
        return str(output_data) 