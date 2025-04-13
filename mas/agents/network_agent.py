from .base_agent import BaseAgent
from langchain.agents import Tool
from typing import List, Dict, Any
import kubernetes as k8s
import os
from ..utils.prometheus_client import PrometheusClient

class NetworkAgent(BaseAgent):
    """Agent responsible for managing network operations in the Kubernetes cluster"""
    
    def __init__(self, llm):
        """Initialize the network agent"""
        super().__init__(llm)
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        self.prometheus = PrometheusClient(base_url=prometheus_url)
        self.k8s_client = k8s.client.CoreV1Api()
        self.networking_client = k8s.client.NetworkingV1Api()
    
    def get_tools(self) -> List[Tool]:
        """Get the network management tools available to this agent"""
        return [
            Tool(
                name="get_network_metrics",
                func=self._get_network_metrics,
                description="Get network metrics for a pod or service"
            ),
            Tool(
                name="analyze_network_traffic",
                func=self._analyze_network_traffic,
                description="Analyze network traffic patterns"
            ),
            Tool(
                name="check_network_policies",
                func=self._check_network_policies,
                description="Check network policies for a namespace or pod"
            ),
            Tool(
                name="update_network_policy",
                func=self._update_network_policy,
                description="Update network policy rules"
            )
        ]
    
    def _get_network_metrics(self, target: str, namespace: str = "default", 
                           resource_type: str = "pod") -> Dict[str, Any]:
        """Get network metrics for a pod or service"""
        try:
            if resource_type.lower() == "pod":
                # Get pod network metrics from Prometheus
                rx_bytes = self.prometheus.query(
                    f'container_network_receive_bytes_total{{pod="{target}",namespace="{namespace}"}}'
                )
                tx_bytes = self.prometheus.query(
                    f'container_network_transmit_bytes_total{{pod="{target}",namespace="{namespace}"}}'
                )
                rx_packets = self.prometheus.query(
                    f'container_network_receive_packets_total{{pod="{target}",namespace="{namespace}"}}'
                )
                tx_packets = self.prometheus.query(
                    f'container_network_transmit_packets_total{{pod="{target}",namespace="{namespace}"}}'
                )
                
                # Extract values from Prometheus response
                rx_bytes_value = rx_bytes.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if rx_bytes else 0
                tx_bytes_value = tx_bytes.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if tx_bytes else 0
                rx_packets_value = rx_packets.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if rx_packets else 0
                tx_packets_value = tx_packets.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if tx_packets else 0
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'pod',
                    'rx_bytes': float(rx_bytes_value),
                    'tx_bytes': float(tx_bytes_value),
                    'rx_packets': float(rx_packets_value),
                    'tx_packets': float(tx_packets_value)
                }
            elif resource_type.lower() == "service":
                # Get service network metrics
                service = self.k8s_client.read_namespaced_service(target, namespace)
                selector = service.spec.selector
                selector_string = ",".join([f"{k}={v}" for k, v in selector.items()])
                
                # Get metrics for pods matching the service selector
                rx_bytes = self.prometheus.query(
                    f'sum(container_network_receive_bytes_total{{namespace="{namespace}",{selector_string}}})'
                )
                tx_bytes = self.prometheus.query(
                    f'sum(container_network_transmit_bytes_total{{namespace="{namespace}",{selector_string}}})'
                )
                rx_packets = self.prometheus.query(
                    f'sum(container_network_receive_packets_total{{namespace="{namespace}",{selector_string}}})'
                )
                tx_packets = self.prometheus.query(
                    f'sum(container_network_transmit_packets_total{{namespace="{namespace}",{selector_string}}})'
                )
                
                # Extract values from Prometheus response
                rx_bytes_value = rx_bytes.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if rx_bytes else 0
                tx_bytes_value = tx_bytes.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if tx_bytes else 0
                rx_packets_value = rx_packets.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if rx_packets else 0
                tx_packets_value = tx_packets.get('data', {}).get('result', [{}])[0].get('value', [0, 0])[1] if tx_packets else 0
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'service',
                    'rx_bytes': float(rx_bytes_value),
                    'tx_bytes': float(tx_bytes_value),
                    'rx_packets': float(rx_packets_value),
                    'tx_packets': float(tx_packets_value)
                }
            else:
                return {
                    'error': f"Unsupported resource type: {resource_type}"
                }
        except Exception as e:
            return self.handle_error(e)
    
    def _analyze_network_traffic(self, target: str, namespace: str = "default", 
                               resource_type: str = "pod", hours: int = 1) -> Dict[str, Any]:
        """Analyze network traffic patterns"""
        try:
            if resource_type.lower() == "pod":
                # Get pod network metrics for the specified time period
                rx_bytes = self.prometheus.query_range(
                    f'container_network_receive_bytes_total{{pod="{target}",namespace="{namespace}"}}',
                    start=int((datetime.now() - timedelta(hours=hours)).timestamp()),
                    end=int(datetime.now().timestamp()),
                    step="1m"
                )
                tx_bytes = self.prometheus.query_range(
                    f'container_network_transmit_bytes_total{{pod="{target}",namespace="{namespace}"}}',
                    start=int((datetime.now() - timedelta(hours=hours)).timestamp()),
                    end=int(datetime.now().timestamp()),
                    step="1m"
                )
                
                # Extract values from Prometheus response
                rx_trends = []
                tx_trends = []
                
                if rx_bytes and 'data' in rx_bytes and 'result' in rx_bytes['data']:
                    for result in rx_bytes['data']['result']:
                        for value in result.get('values', []):
                            rx_trends.append({
                                'timestamp': value[0],
                                'value': float(value[1])
                            })
                
                if tx_bytes and 'data' in tx_bytes and 'result' in tx_bytes['data']:
                    for result in tx_bytes['data']['result']:
                        for value in result.get('values', []):
                            tx_trends.append({
                                'timestamp': value[0],
                                'value': float(value[1])
                            })
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'pod',
                    'rx_trends': rx_trends,
                    'tx_trends': tx_trends,
                    'hours': hours
                }
            elif resource_type.lower() == "service":
                # Get service pod selector
                service = self.k8s_client.read_namespaced_service(target, namespace)
                selector = service.spec.selector
                selector_string = ",".join([f"{k}={v}" for k, v in selector.items()])
                
                # Get service network metrics for the specified time period
                rx_bytes = self.prometheus.query_range(
                    f'sum(container_network_receive_bytes_total{{namespace="{namespace}",{selector_string}}})',
                    start=int((datetime.now() - timedelta(hours=hours)).timestamp()),
                    end=int(datetime.now().timestamp()),
                    step="1m"
                )
                tx_bytes = self.prometheus.query_range(
                    f'sum(container_network_transmit_bytes_total{{namespace="{namespace}",{selector_string}}})',
                    start=int((datetime.now() - timedelta(hours=hours)).timestamp()),
                    end=int(datetime.now().timestamp()),
                    step="1m"
                )
                
                # Extract values from Prometheus response
                rx_trends = []
                tx_trends = []
                
                if rx_bytes and 'data' in rx_bytes and 'result' in rx_bytes['data']:
                    for result in rx_bytes['data']['result']:
                        for value in result.get('values', []):
                            rx_trends.append({
                                'timestamp': value[0],
                                'value': float(value[1])
                            })
                
                if tx_bytes and 'data' in tx_bytes and 'result' in tx_bytes['data']:
                    for result in tx_bytes['data']['result']:
                        for value in result.get('values', []):
                            tx_trends.append({
                                'timestamp': value[0],
                                'value': float(value[1])
                            })
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'service',
                    'rx_trends': rx_trends,
                    'tx_trends': tx_trends,
                    'hours': hours
                }
            else:
                return {
                    'error': f"Unsupported resource type: {resource_type}"
                }
        except Exception as e:
            return self.handle_error(e)
    
    def _check_network_policies(self, target: str, namespace: str = "default", 
                              resource_type: str = "namespace") -> Dict[str, Any]:
        """Check network policies for a namespace or pod"""
        try:
            if resource_type.lower() == "namespace":
                # Get all network policies in the namespace
                policies = self.networking_client.list_namespaced_network_policy(namespace)
                
                # Format policy information
                policy_info = []
                for policy in policies.items:
                    policy_info.append({
                        'name': policy.metadata.name,
                        'pod_selector': policy.spec.pod_selector.match_labels if policy.spec.pod_selector else {},
                        'ingress_rules': [
                            {
                                'from': [{
                                    'pod_selector': rule.from_[0].pod_selector.match_labels if rule.from_ and rule.from_[0].pod_selector else {},
                                    'namespace_selector': rule.from_[0].namespace_selector.match_labels if rule.from_ and rule.from_[0].namespace_selector else {}
                                }] if rule.from_ else [],
                                'ports': [{
                                    'protocol': port.protocol,
                                    'port': port.port
                                } for port in rule.ports] if rule.ports else []
                            }
                            for rule in policy.spec.ingress
                        ] if policy.spec.ingress else [],
                        'egress_rules': [
                            {
                                'to': [{
                                    'pod_selector': rule.to[0].pod_selector.match_labels if rule.to and rule.to[0].pod_selector else {},
                                    'namespace_selector': rule.to[0].namespace_selector.match_labels if rule.to and rule.to[0].namespace_selector else {}
                                }] if rule.to else [],
                                'ports': [{
                                    'protocol': port.protocol,
                                    'port': port.port
                                } for port in rule.ports] if rule.ports else []
                            }
                            for rule in policy.spec.egress
                        ] if policy.spec.egress else []
                    })
                
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'namespace',
                    'policies': policy_info,
                    'count': len(policy_info)
                }
            elif resource_type.lower() == "pod":
                # Get pod's namespace
                pod = self.k8s_client.read_namespaced_pod(target, namespace)
                pod_labels = pod.metadata.labels
            
                # Get all network policies in the namespace
                policies = self.networking_client.list_namespaced_network_policy(namespace)
            
                # Find policies that match the pod
                matching_policies = []
                for policy in policies.items:
                    policy_selector = policy.spec.pod_selector.match_labels if policy.spec.pod_selector else {}
                    if all(pod_labels.get(k) == v for k, v in policy_selector.items()):
                        matching_policies.append({
                            'name': policy.metadata.name,
                            'pod_selector': policy_selector,
                            'ingress_rules': [
                                {
                                    'from': [{
                                        'pod_selector': rule.from_[0].pod_selector.match_labels if rule.from_ and rule.from_[0].pod_selector else {},
                                        'namespace_selector': rule.from_[0].namespace_selector.match_labels if rule.from_ and rule.from_[0].namespace_selector else {}
                                    }] if rule.from_ else [],
                                    'ports': [{
                                        'protocol': port.protocol,
                                        'port': port.port
                                    } for port in rule.ports] if rule.ports else []
                                }
                                for rule in policy.spec.ingress
                            ] if policy.spec.ingress else [],
                            'egress_rules': [
                                {
                                    'to': [{
                                        'pod_selector': rule.to[0].pod_selector.match_labels if rule.to and rule.to[0].pod_selector else {},
                                        'namespace_selector': rule.to[0].namespace_selector.match_labels if rule.to and rule.to[0].namespace_selector else {}
                                    }] if rule.to else [],
                                    'ports': [{
                                        'protocol': port.protocol,
                                        'port': port.port
                                    } for port in rule.ports] if rule.ports else []
                                }
                                for rule in policy.spec.egress
                            ] if policy.spec.egress else []
                        })
            
                return {
                    'target': target,
                    'namespace': namespace,
                    'resource_type': 'pod',
                    'matching_policies': matching_policies,
                    'count': len(matching_policies)
                }
            else:
                return {
                    'error': f"Unsupported resource type: {resource_type}"
                }
        except Exception as e:
            return self.handle_error(e)
    
    def _update_network_policy(self, policy_name: str, namespace: str = "default", 
                             pod_selector: Dict[str, str] = None,
                             ingress_rules: List[Dict[str, Any]] = None,
                             egress_rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update network policy rules"""
        try:
            # Create or update network policy
            policy = k8s.client.V1NetworkPolicy(
                metadata=k8s.client.V1ObjectMeta(name=policy_name),
                spec=k8s.client.V1NetworkPolicySpec(
                    pod_selector=k8s.client.V1LabelSelector(
                        match_labels=pod_selector or {}
                    ),
                    ingress=[
                        k8s.client.V1NetworkPolicyIngressRule(
                            from_=[
                                k8s.client.V1NetworkPolicyPeer(
                                    pod_selector=k8s.client.V1LabelSelector(
                                        match_labels=rule.get('from', [{}])[0].get('pod_selector', {})
                                    ) if rule.get('from') else None,
                                    namespace_selector=k8s.client.V1LabelSelector(
                                        match_labels=rule.get('from', [{}])[0].get('namespace_selector', {})
                                    ) if rule.get('from') else None
                                )
                                for rule in (ingress_rules or [])
                            ],
                            ports=[
                                k8s.client.V1NetworkPolicyPort(
                                    protocol=port.get('protocol'),
                                    port=port.get('port')
                                )
                                for port in rule.get('ports', [])
                            ] if rule.get('ports') else None
                        )
                        for rule in (ingress_rules or [])
                    ] if ingress_rules else None,
                    egress=[
                        k8s.client.V1NetworkPolicyEgressRule(
                            to=[
                                k8s.client.V1NetworkPolicyPeer(
                                    pod_selector=k8s.client.V1LabelSelector(
                                        match_labels=rule.get('to', [{}])[0].get('pod_selector', {})
                                    ) if rule.get('to') else None,
                                    namespace_selector=k8s.client.V1LabelSelector(
                                        match_labels=rule.get('to', [{}])[0].get('namespace_selector', {})
                                    ) if rule.get('to') else None
                                )
                                for rule in (egress_rules or [])
                            ],
                            ports=[
                                k8s.client.V1NetworkPolicyPort(
                                    protocol=port.get('protocol'),
                                    port=port.get('port')
                                )
                                for port in rule.get('ports', [])
                            ] if rule.get('ports') else None
                        )
                        for rule in (egress_rules or [])
                    ] if egress_rules else None
                )
            )
            
            try:
                # Try to update existing policy
                self.networking_client.patch_namespaced_network_policy(policy_name, namespace, policy)
                action = "updated"
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    # Policy doesn't exist, create it
                    self.networking_client.create_namespaced_network_policy(namespace, policy)
                    action = "created"
                else:
                    raise
            
            return {
                'policy_name': policy_name,
                'namespace': namespace,
                'action': action,
                'message': f"Network policy {policy_name} {action} in namespace {namespace}"
            }
        except Exception as e:
            return self.handle_error(e)
    
    def process_result(self, result: str) -> Dict[str, Any]:
        """Process the result from network management actions"""
        # TODO: Implement result processing logic
        return {
            'status': 'processed',
            'result': result
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for network management"""
        required_fields = ['action', 'target']
        return all(field in input_data for field in required_fields)
    
    def format_output(self, output_data: Dict[str, Any]) -> str:
        """Format the network management output data"""
        # TODO: Implement output formatting logic
        return str(output_data) 