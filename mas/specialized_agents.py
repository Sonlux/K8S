from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from kubernetes import client
import time

class ResourceExhaustionAgent(BaseAgent):
    def __init__(self, k8s_api: client.CoreV1Api, k8s_apps_api: client.AppsV1Api):
        super().__init__("resource-exhaustion", k8s_api, k8s_apps_api)
        self.cpu_threshold = 80.0
        self.memory_threshold = 80.0
        self.max_scale_factor = 2.0
        self.min_scale_factor = 0.5
        self.prediction_window = 30  # minutes
        self.historical_data = {}
        self.scaling_cooldown = 300  # seconds

    def can_handle(self, metrics: Dict[str, Any]) -> bool:
        cpu_usage = float(metrics.get('CPU Usage (%)', 0.0))
        memory_usage = float(metrics.get('Memory Usage (%)', 0.0))
        return cpu_usage > self.cpu_threshold or memory_usage > self.memory_threshold

    def analyze(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        cpu_usage = float(metrics.get('CPU Usage (%)', 0.0))
        memory_usage = float(metrics.get('Memory Usage (%)', 0.0))
        
        # Update historical data
        pod_name = metrics.get('pod_name')
        if pod_name:
            if pod_name not in self.historical_data:
                self.historical_data[pod_name] = []
            self.historical_data[pod_name].append({
                'timestamp': time.time(),
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage
            })
            # Keep only last hour of data
            self.historical_data[pod_name] = [
                data for data in self.historical_data[pod_name]
                if time.time() - data['timestamp'] < 3600
            ]
        
        # Calculate current severity
        current_severity = max(
            (cpu_usage - self.cpu_threshold) / 100 if cpu_usage > self.cpu_threshold else 0,
            (memory_usage - self.memory_threshold) / 100 if memory_usage > self.memory_threshold else 0
        )
        
        # Predict future usage
        predicted_severity = self._predict_future_usage(pod_name) if pod_name else current_severity
        
        # Determine if scaling is needed
        needs_scaling = current_severity > 0.1 or predicted_severity > 0.2
        
        # Calculate optimal scale factor
        scale_factor = self._calculate_scale_factor(current_severity, predicted_severity)
        
        return {
            'severity': current_severity,
            'predicted_severity': predicted_severity,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'needs_scaling': needs_scaling,
            'scale_factor': scale_factor,
            'historical_trend': self._get_historical_trend(pod_name) if pod_name else None
        }

    def _predict_future_usage(self, pod_name: str) -> float:
        """Predict future resource usage based on historical data"""
        if pod_name not in self.historical_data or len(self.historical_data[pod_name]) < 2:
            return 0.0
            
        # Simple linear regression for prediction
        data = self.historical_data[pod_name]
        timestamps = [d['timestamp'] for d in data]
        cpu_usages = [d['cpu_usage'] for d in data]
        memory_usages = [d['memory_usage'] for d in data]
        
        # Calculate trends
        cpu_trend = self._calculate_trend(timestamps, cpu_usages)
        memory_trend = self._calculate_trend(timestamps, memory_usages)
        
        # Predict future usage
        future_time = time.time() + (self.prediction_window * 60)
        predicted_cpu = cpu_usages[-1] + (cpu_trend * (future_time - timestamps[-1]))
        predicted_memory = memory_usages[-1] + (memory_trend * (future_time - timestamps[-1]))
        
        return max(
            (predicted_cpu - self.cpu_threshold) / 100 if predicted_cpu > self.cpu_threshold else 0,
            (predicted_memory - self.memory_threshold) / 100 if predicted_memory > self.memory_threshold else 0
        )

    def _calculate_trend(self, timestamps: List[float], values: List[float]) -> float:
        """Calculate the trend of values over time"""
        if len(timestamps) < 2:
            return 0.0
            
        # Simple linear regression
        x_mean = sum(timestamps) / len(timestamps)
        y_mean = sum(values) / len(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(timestamps, values))
        denominator = sum((x - x_mean) ** 2 for x in timestamps)
        
        return numerator / denominator if denominator != 0 else 0.0

    def _calculate_scale_factor(self, current_severity: float, predicted_severity: float) -> float:
        """Calculate optimal scale factor based on current and predicted severity"""
        # Base scale factor on both current and predicted severity
        base_factor = max(current_severity, predicted_severity)
        
        # Apply smoothing to prevent rapid scaling
        scale_factor = min(1.0 + base_factor, self.max_scale_factor)
        
        # Ensure we don't scale below minimum
        return max(scale_factor, self.min_scale_factor)

    def _get_historical_trend(self, pod_name: str) -> Dict[str, Any]:
        """Get historical usage trend for a pod"""
        if pod_name not in self.historical_data:
            return None
            
        data = self.historical_data[pod_name]
        if len(data) < 2:
            return None
            
        # Calculate trends
        timestamps = [d['timestamp'] for d in data]
        cpu_trend = self._calculate_trend(timestamps, [d['cpu_usage'] for d in data])
        memory_trend = self._calculate_trend(timestamps, [d['memory_usage'] for d in data])
        
        return {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'trend_direction': 'increasing' if (cpu_trend + memory_trend) / 2 > 0 else 'decreasing'
        }

    def remediate(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        if not analysis.get('needs_scaling'):
            return {'action_taken': False, 'reason': 'No scaling needed'}

        try:
            pod = self.k8s_api.read_namespaced_pod(
                name=analysis['pod_name'],
                namespace=analysis['namespace']
            )
            
            # Find the deployment
            deployment_name = self._get_deployment_name(pod)
            if not deployment_name:
                return {'action_taken': False, 'reason': 'No deployment found'}

            # Check scaling cooldown
            last_scale_time = self._get_last_scale_time(deployment_name)
            if last_scale_time and time.time() - last_scale_time < self.scaling_cooldown:
                return {'action_taken': False, 'reason': 'Scaling cooldown in effect'}

            # Scale the deployment
            deployment = self.k8s_apps_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=analysis['namespace']
            )
            
            current_replicas = deployment.spec.replicas or 1
            new_replicas = min(int(current_replicas * analysis['scale_factor']), 10)
            
            if new_replicas > current_replicas:
                deployment.spec.replicas = new_replicas
                self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=analysis['namespace'],
                    body=deployment
                )
                self._update_last_scale_time(deployment_name)
                self.record_action('scale_deployment', True, 
                    f"Scaled {deployment_name} from {current_replicas} to {new_replicas} replicas")
                return {'action_taken': True, 'action': 'scale_deployment', 
                       'details': f"Scaled to {new_replicas} replicas"}
            
            return {'action_taken': False, 'reason': 'No scaling needed'}
            
        except Exception as e:
            self.logger.error(f"Error scaling deployment: {str(e)}")
            return {'action_taken': False, 'error': str(e)}

    def _get_last_scale_time(self, deployment_name: str) -> Optional[float]:
        """Get the timestamp of the last scaling action for a deployment"""
        return getattr(self, f'_last_scale_{deployment_name}', None)

    def _update_last_scale_time(self, deployment_name: str) -> None:
        """Update the timestamp of the last scaling action for a deployment"""
        setattr(self, f'_last_scale_{deployment_name}', time.time())

    def _get_deployment_name(self, pod) -> Optional[str]:
        owner_refs = pod.metadata.owner_references or []
        for ref in owner_refs:
            if ref.kind == 'ReplicaSet':
                rs = self.k8s_apps_api.read_namespaced_replica_set(
                    name=ref.name,
                    namespace=pod.metadata.namespace
                )
                rs_owner_refs = rs.metadata.owner_references or []
                for rs_ref in rs_owner_refs:
                    if rs_ref.kind == 'Deployment':
                        return rs_ref.name
        return None

class CrashLoopAgent(BaseAgent):
    def __init__(self, k8s_api: client.CoreV1Api, k8s_apps_api: client.AppsV1Api):
        super().__init__("crash-loop", k8s_api, k8s_apps_api)
        self.restart_threshold = 5
        self.max_restarts = 10

    def can_handle(self, metrics: Dict[str, Any]) -> bool:
        restarts = int(metrics.get('Pod Restarts', 0))
        return restarts >= self.restart_threshold

    def analyze(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        restarts = int(metrics.get('Pod Restarts', 0))
        severity = min(restarts / self.max_restarts, 1.0)
        
        return {
            'severity': severity,
            'restarts': restarts,
            'needs_action': restarts >= self.restart_threshold,
            'should_delete': restarts >= self.max_restarts
        }

    def remediate(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        if not analysis.get('needs_action'):
            return {'action_taken': False, 'reason': 'No action needed'}

        try:
            if analysis.get('should_delete'):
                # Delete the pod if it has exceeded max restarts
                self.k8s_api.delete_namespaced_pod(
                    name=analysis['pod_name'],
                    namespace=analysis['namespace']
                )
                self.record_action('delete_pod', True, 
                    f"Deleted pod {analysis['pod_name']} due to excessive restarts")
                return {'action_taken': True, 'action': 'delete_pod'}
            else:
                # Get pod logs for analysis
                logs = self.k8s_api.read_namespaced_pod_log(
                    name=analysis['pod_name'],
                    namespace=analysis['namespace']
                )
                # TODO: Add log analysis logic here
                return {'action_taken': False, 'reason': 'Log analysis needed'}
                
        except Exception as e:
            self.logger.error(f"Error handling crash loop: {str(e)}")
            return {'action_taken': False, 'error': str(e)}

class NetworkIssueAgent(BaseAgent):
    def __init__(self, k8s_api: client.CoreV1Api, k8s_apps_api: client.AppsV1Api):
        super().__init__("network-issue", k8s_api, k8s_apps_api)
        self.packet_loss_threshold = 5.0  # percentage
        self.connection_timeout = 5  # seconds
        self.max_retries = 3
        self.retry_delay = 10  # seconds
        self.network_policy_cache = {}
        self.service_endpoints_cache = {}
        self.last_diagnostics = {}

    def can_handle(self, metrics: Dict[str, Any]) -> bool:
        network_errors = metrics.get('network_errors', 0)
        packet_loss = metrics.get('packet_loss', 0.0)
        return network_errors > 0 or packet_loss > self.packet_loss_threshold

    def analyze(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        pod_name = metrics.get('pod_name')
        namespace = metrics.get('namespace')
        
        if not pod_name or not namespace:
            return {'severity': 0, 'action_needed': False}

        # Perform comprehensive network diagnostics
        diagnostics = self._perform_network_diagnostics(pod_name, namespace)
        self.last_diagnostics[pod_name] = diagnostics

        # Calculate severity based on multiple factors
        severity = self._calculate_severity(diagnostics)
        
        # Determine if action is needed
        action_needed = severity > 0.3 or diagnostics.get('critical_issues', False)

        return {
            'severity': severity,
            'action_needed': action_needed,
            'diagnostics': diagnostics,
            'recommended_actions': self._get_recommended_actions(diagnostics)
        }

    def _perform_network_diagnostics(self, pod_name: str, namespace: str) -> Dict[str, Any]:
        """Perform comprehensive network diagnostics for a pod"""
        diagnostics = {
            'pod_name': pod_name,
            'namespace': namespace,
            'timestamp': time.time(),
            'issues': [],
            'critical_issues': False
        }

        try:
            # Get pod details
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Check pod network status
            if not pod.status.pod_ip:
                diagnostics['issues'].append('Pod has no IP address assigned')
                diagnostics['critical_issues'] = True

            # Check network policies
            network_policies = self._get_network_policies(namespace)
            if network_policies:
                policy_issues = self._check_network_policies(pod, network_policies)
                diagnostics['network_policy_issues'] = policy_issues
                if policy_issues:
                    diagnostics['issues'].append('Network policy conflicts detected')

            # Check service endpoints
            service_endpoints = self._get_service_endpoints(namespace)
            if service_endpoints:
                endpoint_issues = self._check_service_endpoints(pod, service_endpoints)
                diagnostics['endpoint_issues'] = endpoint_issues
                if endpoint_issues:
                    diagnostics['issues'].append('Service endpoint connectivity issues')

            # Check DNS resolution
            dns_issues = self._check_dns_resolution(pod)
            if dns_issues:
                diagnostics['dns_issues'] = dns_issues
                diagnostics['issues'].append('DNS resolution issues')

            # Check node network status
            node_issues = self._check_node_network(pod.spec.node_name)
            if node_issues:
                diagnostics['node_network_issues'] = node_issues
                diagnostics['issues'].append('Node network issues detected')

        except Exception as e:
            diagnostics['issues'].append(f'Error performing diagnostics: {str(e)}')
            diagnostics['critical_issues'] = True

        return diagnostics

    def _calculate_severity(self, diagnostics: Dict[str, Any]) -> float:
        """Calculate severity score based on diagnostics"""
        severity = 0.0
        
        # Weight different types of issues
        weights = {
            'critical_issues': 1.0,
            'network_policy_issues': 0.7,
            'endpoint_issues': 0.6,
            'dns_issues': 0.5,
            'node_network_issues': 0.8
        }

        for issue_type, weight in weights.items():
            if diagnostics.get(issue_type):
                severity += weight

        return min(severity, 1.0)

    def _get_recommended_actions(self, diagnostics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommended actions based on diagnostics"""
        actions = []

        if diagnostics.get('critical_issues'):
            actions.append({
                'action': 'restart_pod',
                'priority': 'high',
                'reason': 'Critical network issues detected'
            })

        if diagnostics.get('network_policy_issues'):
            actions.append({
                'action': 'update_network_policy',
                'priority': 'medium',
                'reason': 'Network policy conflicts need resolution'
            })

        if diagnostics.get('dns_issues'):
            actions.append({
                'action': 'check_dns_config',
                'priority': 'medium',
                'reason': 'DNS resolution issues detected'
            })

        if diagnostics.get('node_network_issues'):
            actions.append({
                'action': 'check_node_network',
                'priority': 'high',
                'reason': 'Node network issues detected'
            })

        return actions

    def remediate(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        if not analysis.get('action_needed'):
            return {'action_taken': False, 'reason': 'No action needed'}

        diagnostics = analysis.get('diagnostics', {})
        pod_name = diagnostics.get('pod_name')
        namespace = diagnostics.get('namespace')

        if not pod_name or not namespace:
            return {'action_taken': False, 'reason': 'Missing pod information'}

        try:
            # Get recommended actions
            actions = analysis.get('recommended_actions', [])
            
            # Sort actions by priority
            actions.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])

            for action in actions:
                if action['action'] == 'restart_pod':
                    result = self._restart_pod(pod_name, namespace)
                    if result['action_taken']:
                        return result
                elif action['action'] == 'update_network_policy':
                    result = self._update_network_policy(pod_name, namespace)
                    if result['action_taken']:
                        return result
                elif action['action'] == 'check_dns_config':
                    result = self._check_dns_config(pod_name, namespace)
                    if result['action_taken']:
                        return result
                elif action['action'] == 'check_node_network':
                    result = self._check_node_network(pod_name, namespace)
                    if result['action_taken']:
                        return result

            return {'action_taken': False, 'reason': 'No successful remediation actions'}

        except Exception as e:
            self.logger.error(f"Error during remediation: {str(e)}")
            return {'action_taken': False, 'error': str(e)}

    def _restart_pod(self, pod_name: str, namespace: str) -> Dict[str, Any]:
        """Restart a pod with proper checks and retries"""
        for attempt in range(self.max_retries):
            try:
                # Delete the pod (Kubernetes will recreate it)
                self.k8s_api.delete_namespaced_pod(
                    name=pod_name,
                    namespace=namespace,
                    body=client.V1DeleteOptions()
                )
                
                # Wait for pod to be recreated
                time.sleep(self.retry_delay)
                
                # Check if pod is running
                pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
                if pod.status.phase == 'Running':
                    self.record_action('restart_pod', True, f"Successfully restarted pod {pod_name}")
                    return {'action_taken': True, 'action': 'restart_pod'}
                
            except Exception as e:
                self.logger.error(f"Error restarting pod (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue

        return {'action_taken': False, 'reason': 'Failed to restart pod after multiple attempts'}

    def _update_network_policy(self, pod_name: str, namespace: str) -> Dict[str, Any]:
        """Update network policies for a pod"""
        try:
            # Get current network policies
            policies = self._get_network_policies(namespace)
            if not policies:
                return {'action_taken': False, 'reason': 'No network policies found'}

            # Update policies based on diagnostics
            updated = False
            for policy in policies:
                if self._needs_policy_update(policy, pod_name):
                    # Update the policy
                    self.k8s_api.patch_namespaced_network_policy(
                        name=policy.metadata.name,
                        namespace=namespace,
                        body=self._create_updated_policy(policy, pod_name)
                    )
                    updated = True

            if updated:
                self.record_action('update_network_policy', True, 
                    f"Updated network policies for pod {pod_name}")
                return {'action_taken': True, 'action': 'update_network_policy'}

            return {'action_taken': False, 'reason': 'No policy updates needed'}

        except Exception as e:
            self.logger.error(f"Error updating network policy: {str(e)}")
            return {'action_taken': False, 'error': str(e)}

    def _check_dns_config(self, pod_name: str, namespace: str) -> Dict[str, Any]:
        """Check and fix DNS configuration issues"""
        try:
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Check DNS policy
            if pod.spec.dns_policy != 'ClusterFirst':
                # Update pod with correct DNS policy
                pod.spec.dns_policy = 'ClusterFirst'
                self.k8s_api.patch_namespaced_pod(
                    name=pod_name,
                    namespace=namespace,
                    body=pod
                )
                self.record_action('update_dns_config', True, 
                    f"Updated DNS policy for pod {pod_name}")
                return {'action_taken': True, 'action': 'update_dns_config'}

            return {'action_taken': False, 'reason': 'DNS configuration is correct'}

        except Exception as e:
            self.logger.error(f"Error checking DNS config: {str(e)}")
            return {'action_taken': False, 'error': str(e)}

    def _get_network_policies(self, namespace: str) -> List[Any]:
        """Get network policies for a namespace"""
        cache_key = f"{namespace}_policies"
        if cache_key in self.network_policy_cache:
            return self.network_policy_cache[cache_key]

        try:
            policies = self.k8s_api.list_namespaced_network_policy(namespace)
            self.network_policy_cache[cache_key] = policies.items
            return policies.items
        except Exception:
            return []

    def _get_service_endpoints(self, namespace: str) -> List[Any]:
        """Get service endpoints for a namespace"""
        cache_key = f"{namespace}_endpoints"
        if cache_key in self.service_endpoints_cache:
            return self.service_endpoints_cache[cache_key]

        try:
            endpoints = self.k8s_api.list_namespaced_endpoints(namespace)
            self.service_endpoints_cache[cache_key] = endpoints.items
            return endpoints.items
        except Exception:
            return []

    def _check_network_policies(self, pod: Any, policies: List[Any]) -> List[str]:
        """Check if a pod has any network policy issues"""
        issues = []
        pod_labels = pod.metadata.labels or {}
        
        for policy in policies:
            # Check if pod is affected by the policy
            if self._is_pod_affected_by_policy(pod_labels, policy):
                # Check if policy is properly configured
                if not self._is_policy_properly_configured(policy):
                    issues.append(f"Policy {policy.metadata.name} is not properly configured")
                
                # Check for conflicts with other policies
                conflicts = self._check_policy_conflicts(policy, policies)
                if conflicts:
                    issues.extend(conflicts)

        return issues

    def _check_service_endpoints(self, pod: Any, endpoints: List[Any]) -> List[str]:
        """Check if a pod has any service endpoint issues"""
        issues = []
        pod_ip = pod.status.pod_ip

        for endpoint in endpoints:
            # Check if pod is part of the endpoint
            if self._is_pod_in_endpoint(pod_ip, endpoint):
                # Check endpoint configuration
                if not self._is_endpoint_properly_configured(endpoint):
                    issues.append(f"Endpoint {endpoint.metadata.name} is not properly configured")
                
                # Check for connectivity issues
                if not self._check_endpoint_connectivity(endpoint):
                    issues.append(f"Cannot connect to endpoint {endpoint.metadata.name}")

        return issues

    def _check_dns_resolution(self, pod: Any) -> List[str]:
        """Check DNS resolution for a pod"""
        issues = []
        
        # Check if pod has DNS configuration
        if not pod.spec.dns_config:
            issues.append("Pod has no DNS configuration")
            return issues

        # Check DNS policy
        if pod.spec.dns_policy not in ['ClusterFirst', 'ClusterFirstWithHostNet', 'Default']:
            issues.append(f"Invalid DNS policy: {pod.spec.dns_policy}")

        # Check DNS servers
        if pod.spec.dns_config.nameservers:
            for server in pod.spec.dns_config.nameservers:
                if not self._is_valid_ip(server):
                    issues.append(f"Invalid DNS server IP: {server}")

        return issues

    def _check_node_network(self, node_name: str) -> List[str]:
        """Check network status of a node"""
        issues = []
        
        try:
            node = self.k8s_api.read_node(node_name)
            
            # Check node network status
            for condition in node.status.conditions:
                if condition.type == 'NetworkUnavailable' and condition.status == 'True':
                    issues.append("Node network is unavailable")
                elif condition.type == 'Ready' and condition.status != 'True':
                    issues.append("Node is not ready")

            # Check node resources
            if node.status.allocatable:
                if 'ephemeral-storage' in node.status.allocatable:
                    storage = self._parse_quantity(node.status.allocatable['ephemeral-storage'])
                    if storage < 10 * 1024 * 1024 * 1024:  # 10GB
                        issues.append("Node has low storage available")

        except Exception as e:
            issues.append(f"Error checking node network: {str(e)}")

        return issues

    def _is_valid_ip(self, ip: str) -> bool:
        """Check if an IP address is valid"""
        try:
            parts = ip.split('.')
            return len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts)
        except (AttributeError, TypeError, ValueError):
            return False

    def _parse_quantity(self, quantity: str) -> int:
        """Parse Kubernetes quantity string to bytes"""
        try:
            return int(quantity)
        except (ValueError, TypeError):
            return 0

    def _is_pod_affected_by_policy(self, pod_labels: Dict[str, str], policy: Any) -> bool:
        """Check if a pod is affected by a network policy"""
        if not policy.spec.pod_selector:
            return True

        selector = policy.spec.pod_selector.match_labels or {}
        return all(pod_labels.get(k) == v for k, v in selector.items())

    def _is_policy_properly_configured(self, policy: Any) -> bool:
        """Check if a network policy is properly configured"""
        if not policy.spec:
            return False

        # Check ingress rules
        if policy.spec.ingress:
            for rule in policy.spec.ingress:
                if not rule.from_:
                    return False

        # Check egress rules
        if policy.spec.egress:
            for rule in policy.spec.egress:
                if not rule.to:
                    return False

        return True

    def _check_policy_conflicts(self, policy: Any, all_policies: List[Any]) -> List[str]:
        """Check for conflicts between network policies"""
        conflicts = []
        
        for other_policy in all_policies:
            if other_policy.metadata.name == policy.metadata.name:
                continue

            # Check for overlapping selectors
            if self._do_selectors_overlap(policy.spec.pod_selector, other_policy.spec.pod_selector):
                conflicts.append(f"Policy {policy.metadata.name} conflicts with {other_policy.metadata.name}")

        return conflicts

    def _do_selectors_overlap(self, selector1: Any, selector2: Any) -> bool:
        """Check if two selectors overlap"""
        if not selector1 or not selector2:
            return True

        labels1 = selector1.match_labels or {}
        labels2 = selector2.match_labels or {}

        return any(labels1.get(k) == v for k, v in labels2.items())

    def _is_pod_in_endpoint(self, pod_ip: str, endpoint: Any) -> bool:
        """Check if a pod is part of an endpoint"""
        if not endpoint.subsets:
            return False

        for subset in endpoint.subsets:
            if subset.addresses:
                for address in subset.addresses:
                    if address.ip == pod_ip:
                        return True

        return False

    def _is_endpoint_properly_configured(self, endpoint: Any) -> bool:
        """Check if an endpoint is properly configured"""
        if not endpoint.subsets:
            return False

        for subset in endpoint.subsets:
            if not subset.addresses or not subset.ports:
                return False

        return True

    def _check_endpoint_connectivity(self, endpoint: Any) -> bool:
        """Check connectivity to an endpoint"""
        if not endpoint.subsets:
            return False

        for subset in endpoint.subsets:
            if subset.addresses:
                for address in subset.addresses:
                    if subset.ports:
                        for port in subset.ports:
                            try:
                                # Try to connect to the endpoint
                                import socket
                                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                sock.settimeout(self.connection_timeout)
                                result = sock.connect_ex((address.ip, port.port))
                                sock.close()
                                if result == 0:
                                    return True
                            except Exception:
                                continue

        return False

    def _needs_policy_update(self, policy: Any, pod_name: str) -> bool:
        """Check if a network policy needs to be updated"""
        # Check if policy is outdated
        if not policy.metadata.annotations or 'last-updated' not in policy.metadata.annotations:
            return True

        # Check if policy has been updated recently
        last_updated = float(policy.metadata.annotations['last-updated'])
        if time.time() - last_updated > 3600:  # 1 hour
            return True

        return False

    def _create_updated_policy(self, policy: Any, pod_name: str) -> Any:
        """Create an updated version of a network policy"""
        updated_policy = policy
        updated_policy.metadata.annotations = updated_policy.metadata.annotations or {}
        updated_policy.metadata.annotations['last-updated'] = str(time.time())
        return updated_policy 