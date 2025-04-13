from typing import Dict, Any, List, Optional
from kubernetes import client, config
import logging
import time
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
import os
import numpy as np

class Coordinator:
    def __init__(self, llm, tools: List[Any], prompt, memory: ConversationBufferMemory):
        # Setup logging
        self.logger = logging.getLogger("mas-coordinator")
        
        # Setup Kubernetes client
        self._setup_kubernetes()
        
        # Store the tools and LLM
        self.tools = tools
        self.llm = llm
        self.prompt = prompt
        self.memory = memory
        
        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=llm,
            tools=tools,
            memory=memory,
            verbose=True
        )
        
        # Track pod metrics history
        self.pod_metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.sequence_length = 2  # Number of samples needed for analysis
        
        # Track remediation history
        self.remediation_history: Dict[str, List[Dict[str, Any]]] = {}
        self.cooldown_period = 300  # 5 minutes cooldown between remediations

    def _setup_kubernetes(self):
        """Setup Kubernetes client."""
        try:
            # Load kubeconfig
            config.load_kube_config()
            
            # Create API clients
            self.k8s_api = client.CoreV1Api()
            self.k8s_apps_api = client.AppsV1Api()
            self.k8s_metrics_api = client.CustomObjectsApi()
            
            # Test connection
            self.k8s_api.list_namespace()
            self.logger.info("Successfully connected to Kubernetes cluster")
            
            # Set simulation mode to False since we're connected to a real cluster
            self.simulation_mode = False
            
        except Exception as e:
            self.logger.error(f"Failed to setup Kubernetes client: {str(e)}")
            self.logger.error("The system will run in simulation mode without actual Kubernetes access")
            
            # Set a flag to indicate we're in simulation mode
            self.simulation_mode = True
            
            # Create mock API clients for simulation
            self.k8s_api = None
            self.k8s_apps_api = None
            self.k8s_metrics_api = None

    def _get_pod_metrics(self, pod) -> Dict[str, Any]:
        """Get metrics for a pod"""
        try:
            # Get pod metrics using metrics API
            metrics = {
                'pod_name': pod.metadata.name,
                'namespace': pod.metadata.namespace,
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'restart_count': 0,
                'network_errors': 0.0
            }
            
            # Extract container ID from pod status
            if pod.status.container_statuses:
                for container in pod.status.container_statuses:
                    if container.container_id:
                        metrics['container_id'] = container.container_id
                        metrics['restart_count'] = container.restart_count
                        break
            
            # Get metrics from metrics-server if available
            try:
                metrics_data = self.k8s_metrics_api.get_namespaced_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    namespace=pod.metadata.namespace,
                    plural="pods",
                    name=pod.metadata.name
                )
                
                if metrics_data and 'containers' in metrics_data:
                    for container in metrics_data['containers']:
                        if container['name'] == pod.spec.containers[0].name:
                            usage = container.get('usage', {})
                            metrics['cpu_usage'] = self._parse_cpu_usage(usage.get('cpu', '0'))
                            metrics['memory_usage'] = self._parse_memory_usage(usage.get('memory', '0'))
                            break
            except Exception as e:
                self.logger.warning(f"Could not get metrics from metrics-server: {e}")
            
            # Get network metrics if available
            try:
                network_metrics = self.k8s_api.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=pod.metadata.namespace,
                    container=pod.spec.containers[0].name,
                    tail_lines=100
                )
                # Count network errors in logs
                metrics['network_errors'] = network_metrics.lower().count('connection refused') + \
                                          network_metrics.lower().count('timeout') + \
                                          network_metrics.lower().count('network error')
            except Exception as e:
                self.logger.warning(f"Could not get network metrics: {e}")
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting metrics for pod {pod.metadata.name}: {str(e)}")
            return {
                'pod_name': pod.metadata.name,
                'namespace': pod.metadata.namespace,
                'error': str(e)
            }

    def _parse_cpu_usage(self, cpu_str: str) -> float:
        """Parse CPU usage string to percentage"""
        try:
            if not cpu_str:
                return 0.0
            # Convert from nanocores to percentage
            nanocores = int(cpu_str.replace('n', ''))
            return (nanocores / 10000000.0) * 100.0  # Assuming 1 CPU = 1000000000 nanocores
        except:
            return 0.0

    def _parse_memory_usage(self, memory_str: str) -> float:
        """Parse memory usage string to percentage"""
        try:
            if not memory_str:
                return 0.0
            # Convert from bytes to percentage
            bytes_used = self._parse_memory(memory_str)
            return (bytes_used / (1024 * 1024 * 1024)) * 100.0  # Assuming 1GB total memory
        except:
            return 0.0

    def _can_remediate(self, pod_id: str) -> bool:
        """Check if enough time has passed since last remediation"""
        if pod_id not in self.remediation_history:
            return True
            
        last_remediation = self.remediation_history[pod_id][-1]
        time_since_last = time.time() - last_remediation['timestamp']
        return time_since_last >= self.cooldown_period

    def _record_remediation(self, pod_id: str, action: str, success: bool, details: str):
        """Record a remediation action"""
        if pod_id not in self.remediation_history:
            self.remediation_history[pod_id] = []
            
        self.remediation_history[pod_id].append({
            'action': action,
            'success': success,
            'details': details,
            'timestamp': time.time()
        })

    def process_pod(self, pod) -> Dict[str, Any]:
        """Process a single pod and coordinate agent actions"""
        pod_id = f"{pod.metadata.namespace}/{pod.metadata.name}"
        
        # Get current metrics
        metrics = self._get_pod_metrics(pod)
        if not metrics:
            return {'action_taken': False, 'reason': 'No metrics available'}
            
        # Add pod info to metrics
        metrics['pod_name'] = pod.metadata.name
        metrics['namespace'] = pod.metadata.namespace
        
        # Update metrics history
        if pod_id not in self.pod_metrics_history:
            self.pod_metrics_history[pod_id] = []
            
        self.pod_metrics_history[pod_id].append(metrics)
        
        # Keep only the last N samples
        if len(self.pod_metrics_history[pod_id]) > self.sequence_length:
            self.pod_metrics_history[pod_id] = self.pod_metrics_history[pod_id][-self.sequence_length:]
            
        # Check if we have enough samples for analysis
        if len(self.pod_metrics_history[pod_id]) < self.sequence_length:
            return {'action_taken': False, 'reason': 'Insufficient metrics history'}
            
        # Analyze metrics
        issues = self.analyze_metrics(metrics)
        
        # Check if remediation is needed and allowed
        if issues and self._can_remediate(pod_id):
            # Create a remediation plan
            plan = {
                'pod_id': pod_id,
                'issues': issues,
                'actions': []
            }
            
            # Execute the remediation plan
            success = self.execute_remediation_plan(plan)
            
            # Record the remediation
            self._record_remediation(
                pod_id=pod_id,
                action="remediation",
                success=success,
                details=f"Remediation for issues: {', '.join([issue['type'] for issue in issues])}"
            )
            
            # Add a delay after remediation to prevent overwhelming the cluster
            time.sleep(2)
            
            return {
                'action_taken': True,
                'success': success,
                'issues': issues
            }
            
        # Check if we're in cooldown period
        if pod_id in self.remediation_history:
            last_remediation = self.remediation_history[pod_id][-1]
            time_since_last = time.time() - last_remediation['timestamp']
            if time_since_last < self.cooldown_period:
                remaining = int(self.cooldown_period - time_since_last)
                return {
                    'action_taken': False, 
                    'reason': f'In cooldown period ({remaining}s remaining)',
                    'last_action': last_remediation['action'],
                    'last_action_time': last_remediation['timestamp']
                }
            
        return {'action_taken': False, 'reason': 'No remediation needed'}

    def monitor_cluster(self) -> Dict[str, Any]:
        """Monitor the cluster for issues"""
        try:
            # Check if we're in simulation mode
            if self.simulation_mode:
                self.logger.info("Running in simulation mode - no actual Kubernetes access")
                return {
                    'pods_processed': 0,
                    'actions_taken': 0,
                    'results': [],
                    'simulation_mode': True
                }
            
            # Get all pods
            pods = self.k8s_api.list_pod_for_all_namespaces()
            
            # Process each pod
            results = []
            for pod in pods.items:
                result = self.process_pod(pod)
                if result.get('action_taken', False):
                    results.append({
                        'pod_name': pod.metadata.name,
                        'namespace': pod.metadata.namespace,
                        'result': result
                    })
            
            return {
                'pods_processed': len(pods.items),
                'actions_taken': len(results),
                'results': results
            }
        except Exception as e:
            self.logger.error(f"Error monitoring cluster: {str(e)}")
            return {'error': str(e)}

    def coordinate_remediation(self, issues: List[Dict[str, Any]]) -> None:
        """Coordinate remediation actions for detected issues"""
        for issue in issues:
            # Determine which agent should handle this issue
            # This is a simplified example - in a real system, you would have more sophisticated logic
            if issue.get('type') == 'resource_exhaustion':
                # Use the resource agent
                pass
            elif issue.get('type') == 'crash_loop':
                # Use the remediation agent
                pass
            elif issue.get('type') == 'network_issue':
                # Use the network agent
                pass
            else:
                # Use the analysis agent to determine the best course of action
                pass

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get the current status of the cluster"""
        try:
            # Check if we're in simulation mode
            if self.simulation_mode:
                return {
                    'nodes': 0,
                    'pods': 0,
                    'pod_statuses': {},
                    'simulation_mode': True
                }
            
            # Get cluster metrics
            nodes = self.k8s_api.list_node()
            pods = self.k8s_api.list_pod_for_all_namespaces()
            
            # Count pods by status
            pod_statuses = {}
            for pod in pods.items:
                status = pod.status.phase
                if status not in pod_statuses:
                    pod_statuses[status] = 0
                pod_statuses[status] += 1
            
            return {
                'nodes': len(nodes.items),
                'pods': len(pods.items),
                'pod_statuses': pod_statuses
            }
        except Exception as e:
            self.logger.error(f"Error getting cluster status: {str(e)}")
            return {'error': str(e)}

    def analyze_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze metrics and determine if remediation is needed."""
        self.logger.info("Analyzing metrics")
        
        # Get the pod name and namespace
        pod_name = metrics.get("pod_name")
        namespace = metrics.get("namespace")
        
        if not pod_name or not namespace:
            self.logger.error("Missing pod name or namespace in metrics")
            return []
        
        # Check if this is a core Kubernetes component
        is_core_component = namespace == "kube-system" and any(
            component in pod_name for component in [
                "etcd", "kube-apiserver", "kube-controller-manager", "kube-scheduler"
            ]
        )
        
        # Get the container ID
        container_id = metrics.get("container_id")
        if not container_id:
            self.logger.warning(f"Missing container ID for pod {namespace}/{pod_name} - skipping analysis")
            return []
        
        # Check for resource exhaustion
        cpu_usage = metrics.get("cpu_usage", 0)
        memory_usage = metrics.get("memory_usage", 0)
        
        # Use higher thresholds for core components
        cpu_threshold = 90 if is_core_component else 80
        memory_threshold = 90 if is_core_component else 80
        
        if cpu_usage > cpu_threshold or memory_usage > memory_threshold:
            self.logger.warning(f"Resource exhaustion detected for {namespace}/{pod_name}: CPU={cpu_usage}%, Memory={memory_usage}%")
            
            if is_core_component:
                self.logger.info(f"Handling resource issue for core component {namespace}/{pod_name}")
                self._handle_core_component_resource_issue(pod_name, namespace, container_id)
            else:
                self.logger.info(f"Handling resource issue for non-core component {namespace}/{pod_name}")
                self._handle_resource_issue(pod_name, namespace, container_id)
            
            return [{
                "type": "resource_exhaustion",
                "pod_name": pod_name,
                "namespace": namespace,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage
            }]
        
        # Check for crash loops
        restart_count = metrics.get("restart_count", 0)
        
        # Use higher thresholds for core components
        restart_threshold = 10 if is_core_component else 5
        
        if restart_count > restart_threshold:
            self.logger.warning(f"Crash loop detected for {namespace}/{pod_name}: {restart_count} restarts")
            
            if is_core_component:
                self.logger.info(f"Handling crash loop for core component {namespace}/{pod_name}")
                self._handle_core_component_crash_loop(pod_name, namespace, container_id)
            else:
                self.logger.info(f"Handling crash loop for non-core component {namespace}/{pod_name}")
                self._handle_crash_loop(pod_name, namespace, container_id)
            
            return [{
                "type": "crash_loop",
                "pod_name": pod_name,
                "namespace": namespace,
                "restart_count": restart_count
            }]
        
        # Check for network issues
        network_errors = metrics.get("network_errors", 0)
        
        # Use higher thresholds for core components
        network_threshold = 30 if is_core_component else 20
        
        if network_errors > network_threshold:
            self.logger.warning(f"Network issues detected for {namespace}/{pod_name}: {network_errors} errors")
            
            if is_core_component:
                self.logger.info(f"Handling network issue for core component {namespace}/{pod_name}")
                self._handle_core_component_network_issue(pod_name, namespace, container_id)
            else:
                self.logger.info(f"Handling network issue for non-core component {namespace}/{pod_name}")
                self._handle_network_issue(pod_name, namespace, container_id)
            
            return [{
                "type": "network_issue",
                "pod_name": pod_name,
                "namespace": namespace,
                "network_errors": network_errors
            }]
        
        # No issues detected
        self.logger.info(f"No issues detected for {namespace}/{pod_name}")
        return []

    def execute_remediation_plan(self, plan: Dict[str, Any]) -> bool:
        """Execute a remediation plan"""
        try:
            # Check if we're in simulation mode
            if self.simulation_mode:
                self.logger.info(f"Simulating remediation plan: {plan}")
                return True
            
            # This is a simplified example - in a real system, you would have more sophisticated logic
            pod_id = plan.get('pod_id')
            issues = plan.get('issues', [])
            
            if not pod_id or not issues:
                return False
            
            # Extract pod name and namespace
            namespace, pod_name = pod_id.split('/')
            
            # Log the start of remediation
            self.logger.info(f"Starting remediation for pod {pod_id}")
            
            # Check if this is a core Kubernetes component
            is_core_component = namespace == 'kube-system' and any(
                component in pod_name for component in [
                    'etcd', 'kube-apiserver', 'kube-controller-manager', 
                    'kube-scheduler', 'kube-proxy', 'coredns'
                ]
            )
            
            if is_core_component:
                self.logger.warning(f"Detected core Kubernetes component: {pod_id}")
                self.logger.warning("Using special handling for core components")
                
                # For core components, we need to be more careful
                # Instead of restarting, we'll try to diagnose the issue first
                try:
                    # Get pod details
                    pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
                    
                    # Check container status
                    if pod.status.container_statuses:
                        for container in pod.status.container_statuses:
                            # Get container logs to understand the crash
                            try:
                                logs = self.k8s_api.read_namespaced_pod_log(
                                    name=pod_name,
                                    namespace=namespace,
                                    container=container.name,
                                    tail_lines=100
                                )
                                self.logger.info(f"Core component logs: {logs[:1000]}...")
                                
                                # Check for common issues in core components
                                if "connection refused" in logs.lower():
                                    self.logger.warning(f"Connection refused error in {pod_id}. This may indicate network issues.")
                                    # For connection issues, we might need to check network policies
                                    return self._handle_core_component_network_issue(pod_name, namespace, pod_name)
                                
                                elif "out of memory" in logs.lower() or "oom" in logs.lower():
                                    self.logger.warning(f"Out of memory error in {pod_id}. This may indicate resource constraints.")
                                    # For memory issues, we might need to adjust resource limits
                                    return self._handle_core_component_resource_issue(pod_name, namespace, pod_name)
                                
                                elif "permission denied" in logs.lower():
                                    self.logger.warning(f"Permission denied error in {pod_id}. This may indicate security context issues.")
                                    # For permission issues, we might need to check security contexts
                                    return self._handle_core_component_permission_issue(pod_name, namespace, pod_name)
                                
                                else:
                                    # If we can't identify a specific issue, log the error and return
                                    self.logger.error(f"Unidentified issue in core component {pod_id}. Manual intervention may be required.")
                                    return False
                            except Exception as e:
                                self.logger.error(f"Error getting container logs: {str(e)}")
                                return False
                    else:
                        self.logger.warning(f"No container status information for core component {pod_id}")
                        return False
                except Exception as e:
                    self.logger.error(f"Error diagnosing core component {pod_id}: {str(e)}")
                    return False
            
            # For non-core components, proceed with normal remediation
            for issue in issues:
                issue_type = issue.get('type')
                severity = issue.get('severity', 'medium')
                details = issue.get('details', '')
                
                self.logger.info(f"Remediating issue: {issue_type} (severity: {severity}) - {details}")
                
                if issue_type == 'resource_exhaustion':
                    # Get the deployment name
                    deployment_name = self._get_deployment_name(pod_name, namespace)
                    if deployment_name:
                        # Get current replicas
                        try:
                            deployment = self.k8s_apps_api.read_namespaced_deployment(deployment_name, namespace)
                            current_replicas = deployment.spec.replicas
                            
                            # Scale up by 50% (at least 1 more replica)
                            new_replicas = max(current_replicas + 1, int(current_replicas * 1.5))
                            
                            self.logger.info(f"Scaling deployment {deployment_name} from {current_replicas} to {new_replicas} replicas")
                            
                            # Update the deployment
                            self.k8s_apps_api.patch_namespaced_deployment(
                                name=deployment_name,
                                namespace=namespace,
                                body={'spec': {'replicas': new_replicas}}
                            )
                            
                            # Add a delay to allow the scaling to take effect
                            time.sleep(5)
                        except Exception as e:
                            self.logger.error(f"Error scaling deployment {deployment_name}: {str(e)}")
                            return False
                    else:
                        self.logger.warning(f"Could not find deployment for pod {pod_id}")
                        return False
                
                elif issue_type == 'crash_loop':
                    # Get pod details to understand the crash
                    try:
                        pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
                        
                        # Check container status
                        if pod.status.container_statuses:
                            for container in pod.status.container_statuses:
                                if container.state.waiting:
                                    reason = container.state.waiting.reason
                                    message = container.state.waiting.message
                                    self.logger.info(f"Container {container.name} is waiting: {reason} - {message}")
                                    
                                    # Check for common crash reasons
                                    if reason in ['CrashLoopBackOff', 'Error']:
                                        # Get container logs to understand the crash
                                        try:
                                            logs = self.k8s_api.read_namespaced_pod_log(
                                                name=pod_name,
                                                namespace=namespace,
                                                container=container.name,
                                                tail_lines=50
                                            )
                                            self.logger.info(f"Container logs: {logs[:500]}...")
                                        except Exception as e:
                                            self.logger.error(f"Error getting container logs: {str(e)}")
                                    
                                    # For CrashLoopBackOff, try to restart the pod
                                    if reason == 'CrashLoopBackOff':
                                        self.logger.info(f"Restarting pod {pod_id} due to CrashLoopBackOff")
                                        self.k8s_api.delete_namespaced_pod(
                                            name=pod_name,
                                            namespace=namespace
                                        )
                                        
                                        # Add a delay to allow the pod to restart
                                        time.sleep(10)
                                    else:
                                        # For other issues, try to get more information
                                        self.logger.info(f"Pod {pod_id} has issue: {reason} - {message}")
                                        return False
                        else:
                            self.logger.warning(f"No container status information for pod {pod_id}")
                            return False
                    except Exception as e:
                        self.logger.error(f"Error getting pod details: {str(e)}")
                        return False
                
                elif issue_type == 'network_issue':
                    # For network issues, we need to check network policies
                    try:
                        # Get network policies in the namespace
                        network_policies = self.k8s_api.list_namespaced_network_policy(namespace)
                        
                        if network_policies.items:
                            self.logger.info(f"Found {len(network_policies.items)} network policies in namespace {namespace}")
                            
                            # Check if there are restrictive policies
                            for policy in network_policies.items:
                                self.logger.info(f"Network policy {policy.metadata.name} may be affecting pod {pod_id}")
                                
                                # In a real system, we would analyze the policy and make adjustments
                                # For now, we'll just log the policy details
                                self.logger.info(f"Policy details: {policy.spec}")
                        else:
                            self.logger.info(f"No network policies found in namespace {namespace}")
                        
                        # For now, we'll just restart the pod to see if it helps
                        self.logger.info(f"Restarting pod {pod_id} due to network issues")
                        self.k8s_api.delete_namespaced_pod(
                            name=pod_name,
                            namespace=namespace
                        )
                        
                        # Add a delay to allow the pod to restart
                        time.sleep(10)
                    except Exception as e:
                        self.logger.error(f"Error handling network issues: {str(e)}")
                        return False
                
                else:
                    self.logger.warning(f"Unknown issue type: {issue_type}")
                    return False
            
            # Log successful remediation
            self.logger.info(f"Successfully completed remediation for pod {pod_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error executing remediation plan: {str(e)}")
            return False
            
    def _handle_core_component_network_issue(self, pod_name: str, namespace: str, container_id: str) -> bool:
        """Handle network issues for core components."""
        self.logger.info(f"Handling network issue for core component {namespace}/{pod_name}")
        
        # Get the pod
        pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
        
        # Check if the pod is using host network mode
        is_host_network = pod.spec.host_network if pod.spec.host_network else False
        self.logger.info(f"Pod {pod_name} is using host network: {is_host_network}")
        
        # For kube-apiserver, check if it's a metrics API issue
        if "kube-apiserver" in pod_name:
            self.logger.warning("Detected potential metrics API issue with kube-apiserver")
            
            # Check if the metrics-server is running
            try:
                metrics_server_pods = self.k8s_api.list_namespaced_pod(
                    "kube-system",
                    label_selector="k8s-app=metrics-server"
                )
                
                if not metrics_server_pods.items:
                    self.logger.error("metrics-server is not running")
                    self.logger.info("Recommended action: Create or fix metrics-server deployment")
                    self.logger.info("Command: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml")
                    return False
                
                # Check if the metrics API is registered
                try:
                    apiservices = self.k8s_api.list_api_service()
                    metrics_api_found = False
                    for api in apiservices.items:
                        if api.metadata.name == "v1beta1.metrics.k8s.io":
                            metrics_api_found = True
                            self.logger.info(f"Metrics API found: {api.status}")
                            break
                    
                    if not metrics_api_found:
                        self.logger.error("Metrics API is not registered")
                        self.logger.info("Recommended action: Register the metrics API")
                        self.logger.info("Command: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml")
                        return False
                except Exception as e:
                    self.logger.error(f"Error checking API services: {e}")
                    return False
                
                # Check if the metrics-server has the necessary permissions
                try:
                    # Check if the metrics-server service account exists
                    service_account = self.k8s_api.read_namespaced_service_account(
                        "metrics-server",
                        "kube-system"
                    )
                    self.logger.info("metrics-server service account exists")
                    
                    # Check if the service account has the necessary cluster role binding
                    try:
                        cluster_role_binding = self.k8s_api.read_cluster_role_binding("system:metrics-server")
                        self.logger.info("system:metrics-server cluster role binding exists")
                    except Exception as e:
                        self.logger.error("system:metrics-server cluster role binding does not exist")
                        self.logger.info("Recommended action: Create the cluster role binding")
                        self.logger.info("Command: kubectl create clusterrolebinding system:metrics-server --clusterrole=system:metrics-server --serviceaccount=kube-system:metrics-server")
                        return False
                except Exception as e:
                    self.logger.error(f"Error checking metrics-server permissions: {e}")
                    return False
                
                self.logger.info("kube-apiserver appears to have the necessary network configuration for metrics API")
                return False
            except Exception as e:
                self.logger.error(f"Error checking metrics-server: {e}")
                return False
        
        # For etcd, check if it's a disk space issue
        if "etcd" in pod_name:
            self.logger.warning("Detected potential disk space issue with etcd")
            
            # Check if etcd has a persistent volume
            if pod.spec.volumes:
                for volume in pod.spec.volumes:
                    if volume.persistent_volume_claim:
                        pvc_name = volume.persistent_volume_claim.claim_name
                        try:
                            pvc = self.k8s_api.read_namespaced_persistent_volume_claim(pvc_name, namespace)
                            self.logger.info(f"etcd PVC {pvc_name} status: {pvc.status.phase}")
                            
                            # Check if the PVC is bound
                            if pvc.status.phase != "Bound":
                                self.logger.error(f"etcd PVC {pvc_name} is not bound")
                                self.logger.info("Recommended action: Check the PVC and PV status")
                                self.logger.info("Command: kubectl get pvc -n kube-system")
                                self.logger.info("Command: kubectl get pv")
                                return False
                        except Exception as e:
                            self.logger.error(f"Error getting PVC {pvc_name}: {e}")
                            return False
                    elif volume.host_path:
                        self.logger.info(f"etcd is using host path volume: {volume.host_path.path}")
                        
                        # Check if the host path has the correct permissions
                        # This would require access to the host, which we don't have
                        self.logger.warning("Cannot check host path volume permissions")
                        self.logger.info("Recommended action: Check disk space on the node")
                        self.logger.info("Command: df -h")
                        return False
            else:
                self.logger.warning("etcd does not have any volumes")
                self.logger.info("Recommended action: Add a persistent volume to etcd")
                self.logger.info("Command: kubectl edit deployment etcd-minikube -n kube-system")
                return False
        
        # For kube-controller-manager, check if it's a permission issue with the horizontal-pod-autoscaler
        if "kube-controller-manager" in pod_name:
            self.logger.warning("Detected potential permission issue with kube-controller-manager and horizontal-pod-autoscaler")
            
            # Check if the horizontal-pod-autoscaler service account exists
            try:
                service_account = self.k8s_api.read_namespaced_service_account(
                    "horizontal-pod-autoscaler",
                    "kube-system"
                )
                self.logger.info("horizontal-pod-autoscaler service account exists")
                
                # Check if the service account has the necessary role binding
                try:
                    role_bindings = self.k8s_api.list_namespaced_role_binding("kube-system")
                    hpa_role_binding_found = False
                    for binding in role_bindings.items:
                        if "horizontal-pod-autoscaler" in binding.metadata.name:
                            hpa_role_binding_found = True
                            break
                    
                    if not hpa_role_binding_found:
                        self.logger.error("horizontal-pod-autoscaler does not have the necessary role binding")
                        self.logger.info("Recommended action: Create the role binding")
                        self.logger.info("Command: kubectl create rolebinding -n kube-system horizontal-pod-autoscaler --role=system:metrics-server --serviceaccount=kube-system:horizontal-pod-autoscaler")
                        return False
                    
                    self.logger.info("horizontal-pod-autoscaler appears to have the necessary permissions")
                    return False
                except Exception as e:
                    self.logger.error(f"Error checking horizontal-pod-autoscaler permissions: {e}")
                    return False
            except Exception as e:
                self.logger.error(f"Error checking horizontal-pod-autoscaler service account: {e}")
                return False
        
        # For kube-scheduler, check if it's a permission issue with the extension-apiserver-authentication configmap
        if "kube-scheduler" in pod_name:
            self.logger.warning("Detected potential permission issue with kube-scheduler and extension-apiserver-authentication")
            
            # Check if the extension-apiserver-authentication configmap exists
            try:
                configmap = self.k8s_api.read_namespaced_config_map(
                    "extension-apiserver-authentication",
                    "kube-system"
                )
                self.logger.info("extension-apiserver-authentication configmap exists")
                
                # Check if the kube-scheduler has the necessary permissions to access the configmap
                try:
                    # This is a bit tricky to check directly, so we'll just log the configmap
                    self.logger.info(f"extension-apiserver-authentication configmap: {configmap}")
                    
                    # Check if the kube-scheduler service account has the necessary role binding
                    try:
                        role_bindings = self.k8s_api.list_namespaced_role_binding("kube-system")
                        scheduler_role_binding_found = False
                        for binding in role_bindings.items:
                            if "extension-apiserver-authentication-reader" in binding.metadata.name:
                                scheduler_role_binding_found = True
                                break
                        
                        if not scheduler_role_binding_found:
                            self.logger.error("kube-scheduler does not have the necessary role binding for extension-apiserver-authentication")
                            self.logger.info("Recommended action: Create the role binding")
                            self.logger.info("Command: kubectl create rolebinding -n kube-system extension-apiserver-authentication-reader --role=extension-apiserver-authentication-reader --serviceaccount=kube-system:kube-scheduler")
                            return False
                        
                        self.logger.info("kube-scheduler appears to have the necessary permissions for extension-apiserver-authentication")
                        return False
                    except Exception as e:
                        self.logger.error(f"Error checking kube-scheduler role bindings: {e}")
                        return False
                except Exception as e:
                    if "forbidden" in str(e).lower():
                        self.logger.error("kube-scheduler does not have permission to access the extension-apiserver-authentication configmap")
                        self.logger.info("Recommended action: Create the role binding")
                        self.logger.info("Command: kubectl create rolebinding -n kube-system extension-apiserver-authentication-reader --role=extension-apiserver-authentication-reader --serviceaccount=kube-system:kube-scheduler")
                        return False
                    else:
                        self.logger.error(f"Error checking kube-scheduler permissions: {e}")
                        return False
            except Exception as e:
                if "not found" in str(e).lower():
                    self.logger.error("extension-apiserver-authentication configmap does not exist")
                    self.logger.info("Recommended action: Create the configmap")
                    self.logger.info("Command: kubectl create configmap -n kube-system extension-apiserver-authentication --from-literal=client-ca-file=/etc/kubernetes/pki/ca.crt")
                    return False
                else:
                    self.logger.error(f"Error checking extension-apiserver-authentication configmap: {e}")
                    return False
        
        # If we've made it this far, we can try to remediate the crash loop
        self.logger.info(f"Attempting to remediate crash loop for core component {namespace}/{pod_name}")
        
        # Delete the pod to force a restart
        try:
            self.k8s_api.delete_namespaced_pod(pod_name, namespace)
            self.logger.info(f"Deleted pod {namespace}/{pod_name} to force a restart")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting pod {namespace}/{pod_name}: {e}")
            return False 

    def _handle_core_component_resource_issue(self, pod_name: str, namespace: str, container_id: str) -> bool:
        """Handle resource issues for core components."""
        self.logger.info(f"Handling resource issue for core component {namespace}/{pod_name}")
        
        # Get the pod
        pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
        
        # Get the node
        node_name = pod.spec.node_name
        node = self.k8s_api.read_node(node_name)
        
        # Check node allocatable resources
        allocatable = node.status.allocatable
        self.logger.info(f"Node {node_name} allocatable resources: CPU={allocatable['cpu']}, Memory={allocatable['memory']}")
        
        # Check node conditions
        for condition in node.status.conditions:
            if condition.type == "MemoryPressure" and condition.status == "True":
                self.logger.error(f"Node {node_name} is under memory pressure")
                self.logger.info("Recommended action: Check for memory leaks or increase node memory")
                self.logger.info("Command: kubectl top node")
                self.logger.info("Command: kubectl describe node")
                return False
            elif condition.type == "DiskPressure" and condition.status == "True":
                self.logger.error(f"Node {node_name} is under disk pressure")
                self.logger.info("Recommended action: Check for disk space issues or increase node disk space")
                self.logger.info("Command: kubectl describe node")
                self.logger.info("Command: df -h")
                return False
            elif condition.type == "PIDPressure" and condition.status == "True":
                self.logger.error(f"Node {node_name} is under PID pressure")
                self.logger.info("Recommended action: Check for process leaks or increase node PID limit")
                self.logger.info("Command: kubectl describe node")
                self.logger.info("Command: ps aux | wc -l")
                return False
        
        # For etcd, check if it's a disk space issue
        if "etcd" in pod_name:
            self.logger.warning("Detected potential disk space issue with etcd")
            
            # Check if etcd has a persistent volume
            if pod.spec.volumes:
                for volume in pod.spec.volumes:
                    if volume.persistent_volume_claim:
                        pvc_name = volume.persistent_volume_claim.claim_name
                        try:
                            pvc = self.k8s_api.read_namespaced_persistent_volume_claim(pvc_name, namespace)
                            self.logger.info(f"etcd PVC {pvc_name} status: {pvc.status.phase}")
                            
                            # Check if the PVC is bound
                            if pvc.status.phase != "Bound":
                                self.logger.error(f"etcd PVC {pvc_name} is not bound")
                                self.logger.info("Recommended action: Check the PVC and PV status")
                                self.logger.info("Command: kubectl get pvc -n kube-system")
                                self.logger.info("Command: kubectl get pv")
                                return False
                            
                            # Check if the PVC has enough storage
                            if pvc.spec.resources.requests:
                                storage_request = pvc.spec.resources.requests.get("storage")
                                if storage_request:
                                    self.logger.info(f"etcd PVC {pvc_name} storage request: {storage_request}")
                                    
                                    # Check if the storage request is less than 10Gi
                                    if self._parse_storage(storage_request) < 10 * 1024 * 1024 * 1024:  # 10Gi in bytes
                                        self.logger.error(f"etcd PVC {pvc_name} storage request is less than 10Gi")
                                        self.logger.info("Recommended action: Increase the PVC storage request")
                                        self.logger.info("Command: kubectl patch pvc {pvc_name} -n kube-system -p '{\"spec\":{\"resources\":{\"requests\":{\"storage\":\"10Gi\"}}}}'")
                                        return False
                        except Exception as e:
                            self.logger.error(f"Error getting PVC {pvc_name}: {e}")
                            return False
                    elif volume.host_path:
                        self.logger.info(f"etcd is using host path volume: {volume.host_path.path}")
                        
                        # Check if the host path has the correct permissions
                        # This would require access to the host, which we don't have
                        self.logger.warning("Cannot check host path volume permissions")
                        self.logger.info("Recommended action: Check disk space on the node")
                        self.logger.info("Command: df -h")
                        return False
            else:
                self.logger.warning("etcd does not have any volumes")
                self.logger.info("Recommended action: Add a persistent volume to etcd")
                self.logger.info("Command: kubectl edit deployment etcd-minikube -n kube-system")
                return False
        
        # For kube-apiserver, check if it's a memory issue
        if "kube-apiserver" in pod_name:
            self.logger.warning("Detected potential memory issue with kube-apiserver")
            
            # Check if the pod has resource limits
            if pod.spec.containers:
                for container in pod.spec.containers:
                    if container.name == "kube-apiserver":
                        if container.resources.limits:
                            memory_limit = container.resources.limits.get("memory")
                            if memory_limit:
                                self.logger.info(f"kube-apiserver memory limit: {memory_limit}")
                                
                                # Check if the memory limit is less than 1Gi
                                if self._parse_memory(memory_limit) < 1024 * 1024 * 1024:  # 1Gi in bytes
                                    self.logger.error("kube-apiserver memory limit is less than 1Gi")
                                    self.logger.info("Recommended action: Increase the memory limit")
                                    self.logger.info("Command: kubectl patch deployment kube-apiserver-minikube -n kube-system -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"kube-apiserver\",\"resources\":{\"limits\":{\"memory\":\"1Gi\"}}}}]}}}}'")
                                    return False
                        else:
                            self.logger.warning("kube-apiserver does not have resource limits")
                            self.logger.info("Recommended action: Add resource limits")
                            self.logger.info("Command: kubectl edit deployment kube-apiserver-minikube -n kube-system")
                            return False
        
        # For kube-controller-manager, check if it's a CPU issue
        if "kube-controller-manager" in pod_name:
            self.logger.warning("Detected potential CPU issue with kube-controller-manager")
            
            # Check if the pod has resource limits
            if pod.spec.containers:
                for container in pod.spec.containers:
                    if container.name == "kube-controller-manager":
                        if container.resources.limits:
                            cpu_limit = container.resources.limits.get("cpu")
                            if cpu_limit:
                                self.logger.info(f"kube-controller-manager CPU limit: {cpu_limit}")
                                
                                # Check if the CPU limit is less than 200m
                                if self._parse_cpu(cpu_limit) < 0.2:  # 200m in cores
                                    self.logger.error("kube-controller-manager CPU limit is less than 200m")
                                    self.logger.info("Recommended action: Increase the CPU limit")
                                    self.logger.info("Command: kubectl patch deployment kube-controller-manager-minikube -n kube-system -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"kube-controller-manager\",\"resources\":{\"limits\":{\"cpu\":\"200m\"}}}}]}}}}'")
                                    return False
                        else:
                            self.logger.warning("kube-controller-manager does not have resource limits")
                            self.logger.info("Recommended action: Add resource limits")
                            self.logger.info("Command: kubectl edit deployment kube-controller-manager-minikube -n kube-system")
                            return False
        
        # For kube-scheduler, check if it's a general resource issue
        if "kube-scheduler" in pod_name:
            self.logger.warning("Detected potential resource issue with kube-scheduler")
            
            # Check if the pod has resource limits
            if pod.spec.containers:
                for container in pod.spec.containers:
                    if container.name == "kube-scheduler":
                        if container.resources.limits:
                            memory_limit = container.resources.limits.get("memory")
                            cpu_limit = container.resources.limits.get("cpu")
                            
                            if memory_limit:
                                self.logger.info(f"kube-scheduler memory limit: {memory_limit}")
                                
                                # Check if the memory limit is less than 512Mi
                                if self._parse_memory(memory_limit) < 512 * 1024 * 1024:  # 512Mi in bytes
                                    self.logger.error("kube-scheduler memory limit is less than 512Mi")
                                    self.logger.info("Recommended action: Increase the memory limit")
                                    self.logger.info("Command: kubectl patch deployment kube-scheduler-minikube -n kube-system -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"kube-scheduler\",\"resources\":{\"limits\":{\"memory\":\"512Mi\"}}}}]}}}}'")
                                    return False
                            
                            if cpu_limit:
                                self.logger.info(f"kube-scheduler CPU limit: {cpu_limit}")
                                
                                # Check if the CPU limit is less than 100m
                                if self._parse_cpu(cpu_limit) < 0.1:  # 100m in cores
                                    self.logger.error("kube-scheduler CPU limit is less than 100m")
                                    self.logger.info("Recommended action: Increase the CPU limit")
                                    self.logger.info("Command: kubectl patch deployment kube-scheduler-minikube -n kube-system -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"kube-scheduler\",\"resources\":{\"limits\":{\"cpu\":\"100m\"}}}}]}}}}'")
                                    return False
                        else:
                            self.logger.warning("kube-scheduler does not have resource limits")
                            self.logger.info("Recommended action: Add resource limits")
                            self.logger.info("Command: kubectl edit deployment kube-scheduler-minikube -n kube-system")
                            return False
        
        # For all core components, check if the pod has the necessary resource requests
        self.logger.info(f"Core component {namespace}/{pod_name} appears to have the necessary resources")
        return False

    def _handle_resource_issue(self, pod_name: str, namespace: str, container_id: str) -> bool:
        """Handle resource exhaustion for non-core components with smart scaling strategy."""
        self.logger.info(f"Handling resource issue for non-core component {namespace}/{pod_name}")
        
        try:
            # Get the pod
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Get the deployment name
            deployment_name = self._get_deployment_name(pod_name, namespace)
            if not deployment_name:
                self.logger.error(f"Could not determine deployment name for pod {namespace}/{pod_name}")
                return False
            
            # Get the deployment
            deployment = self.k8s_apps_api.read_namespaced_deployment(deployment_name, namespace)
            
            # Get current resource requests and limits
            container = next((c for c in pod.spec.containers if c.name == pod.spec.containers[0].name), None)
            if not container:
                self.logger.error(f"Could not find container in pod {namespace}/{pod_name}")
                return False
            
            # Get current resource requests and limits
            current_cpu_request = container.resources.requests.get('cpu', '100m')
            current_memory_request = container.resources.requests.get('memory', '128Mi')
            current_cpu_limit = container.resources.limits.get('cpu', '200m')
            current_memory_limit = container.resources.limits.get('memory', '256Mi')
            
            # Parse current values
            current_cpu_request_val = self._parse_cpu(current_cpu_request)
            current_memory_request_val = self._parse_memory(current_memory_request)
            current_cpu_limit_val = self._parse_cpu(current_cpu_limit)
            current_memory_limit_val = self._parse_memory(current_memory_limit)
            
            # Get current replicas
            current_replicas = deployment.spec.replicas if deployment.spec.replicas else 1
            
            # Check if HPA is already configured
            try:
                hpa = self.k8s_api.list_namespaced_horizontal_pod_autoscaler(namespace)
                hpa_exists = any(h.metadata.name == deployment_name for h in hpa.items)
            except Exception as e:
                self.logger.warning(f"Error checking for HPA: {e}")
                hpa_exists = False
            
            # Get metrics for the pod
            metrics = self._get_pod_metrics(pod)
            cpu_usage = metrics.get("cpu_usage", 0)
            memory_usage = metrics.get("memory_usage", 0)
            
            # Get pod metrics history
            pod_id = f"{namespace}/{pod_name}"
            
            # PREDICTIVE SCALING: Check historical metrics to predict future resource needs
            if pod_id in self.pod_metrics_history and len(self.pod_metrics_history[pod_id]) >= 3:
                # Get CPU usage trend
                cpu_history = [m.get("cpu_usage", 0) for m in self.pod_metrics_history[pod_id]]
                memory_history = [m.get("memory_usage", 0) for m in self.pod_metrics_history[pod_id]]
                
                # Calculate trend using linear regression
                try:
                    # Calculate CPU trend
                    x = np.arange(len(cpu_history))
                    cpu_slope, cpu_intercept = np.polyfit(x, cpu_history, 1)
                    
                    # Calculate Memory trend
                    memory_slope, memory_intercept = np.polyfit(x, memory_history, 1)
                    
                    # Predict future values (30 minutes ahead)
                    prediction_window = 6  # 6 data points = ~30 min if collection is every 5 min
                    future_x = len(cpu_history) + prediction_window
                    predicted_cpu = cpu_slope * future_x + cpu_intercept
                    predicted_memory = memory_slope * future_x + memory_intercept
                    
                    self.logger.info(f"Current CPU: {cpu_usage:.1f}%, Predicted in 30 mins: {predicted_cpu:.1f}%")
                    self.logger.info(f"Current Memory: {memory_usage:.1f}%, Predicted in 30 mins: {predicted_memory:.1f}%")
                    
                    # Use predicted values if they're higher than current and trend is positive
                    if cpu_slope > 0 and predicted_cpu > cpu_usage:
                        self.logger.info(f"Using predicted CPU usage for scaling decision: {predicted_cpu:.1f}%")
                        cpu_usage = predicted_cpu
                    
                    if memory_slope > 0 and predicted_memory > memory_usage:
                        self.logger.info(f"Using predicted memory usage for scaling decision: {predicted_memory:.1f}%")
                        memory_usage = predicted_memory
                except Exception as e:
                    self.logger.warning(f"Error calculating resource trends: {e}")
            
            # Determine if we should scale horizontally (add replicas)
            should_scale_horizontally = (
                cpu_usage > 80 and  # High CPU usage
                current_replicas < 5 and  # Not too many replicas already
                not hpa_exists  # No HPA configured yet
            )
            
            # Determine if we should scale vertically (increase resources)
            should_scale_vertically = (
                memory_usage > 80 or  # High memory usage
                current_replicas >= 5 or  # Already have many replicas
                hpa_exists  # HPA is configured (better to adjust resources than add replicas)
            )
            
            # Execute scaling strategy
            if should_scale_horizontally and should_scale_vertically:
                # Do both - scale horizontally and vertically
                self.logger.info(f"Scaling {namespace}/{pod_name} both horizontally and vertically")
                
                # Scale horizontally (add replicas)
                new_replicas = min(current_replicas + 2, 10)  # Add 2 replicas, max 10
                deployment.spec.replicas = new_replicas
                
                # Scale vertically (increase resources)
                # Calculate scale factor based on resource usage 
                cpu_scale_factor = 1.0 + (max(0, (cpu_usage - 80)) / 100)
                memory_scale_factor = 1.0 + (max(0, (memory_usage - 80)) / 100)
                
                # Apply scaling with bounds
                new_cpu_request = f"{int(min(current_cpu_request_val * cpu_scale_factor, current_cpu_request_val * 2.0))}m"
                new_memory_request = f"{int(min(current_memory_request_val * memory_scale_factor, current_memory_request_val * 2.0))}Mi"
                new_cpu_limit = f"{int(min(current_cpu_limit_val * cpu_scale_factor, current_cpu_limit_val * 2.0))}m"
                new_memory_limit = f"{int(min(current_memory_limit_val * memory_scale_factor, current_memory_limit_val * 2.0))}Mi"
                
                # Update resources
                deployment.spec.template.spec.containers[0].resources.requests = {
                    'cpu': new_cpu_request,
                    'memory': new_memory_request
                }
                deployment.spec.template.spec.containers[0].resources.limits = {
                    'cpu': new_cpu_limit,
                    'memory': new_memory_limit
                }
                
                # Apply the update
                self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment
                )
                
                self.logger.info(f"Updated {namespace}/{pod_name}: Added replicas (now {new_replicas}) and increased resources")
                self.logger.info(f"New CPU request: {new_cpu_request}, limit: {new_cpu_limit}")
                self.logger.info(f"New memory request: {new_memory_request}, limit: {new_memory_limit}")
                
                # Create HPA for future automatic scaling
                try:
                    hpa_body = {
                        "apiVersion": "autoscaling/v2",
                        "kind": "HorizontalPodAutoscaler",
                        "metadata": {
                            "name": deployment_name,
                            "namespace": namespace
                        },
                        "spec": {
                            "scaleTargetRef": {
                                "apiVersion": "apps/v1",
                                "kind": "Deployment",
                                "name": deployment_name
                            },
                            "minReplicas": max(1, new_replicas - 1),
                            "maxReplicas": 20,
                            "metrics": [
                                {
                                    "type": "Resource",
                                    "resource": {
                                        "name": "cpu",
                                        "target": {
                                            "type": "Utilization",
                                            "averageUtilization": 70
                                        }
                                    }
                                },
                                {
                                    "type": "Resource",
                                    "resource": {
                                        "name": "memory",
                                        "target": {
                                            "type": "Utilization",
                                            "averageUtilization": 70
                                        }
                                    }
                                }
                            ]
                        }
                    }
                    
                    # Try to create the HPA
                    try:
                        self.k8s_api.create_namespaced_horizontal_pod_autoscaler(namespace, hpa_body)
                        self.logger.info(f"Created HPA for {namespace}/{pod_name}")
                    except Exception as e:
                        self.logger.warning(f"Could not create HPA: {e}")
                except Exception as e:
                    self.logger.warning(f"Error creating HPA: {e}")
                
                # Record the remediation
                self._record_remediation(
                    pod_id=f"{namespace}/{pod_name}",
                    action="scaled_horizontally_and_vertically",
                    success=True,
                    details=f"Added replicas (now {new_replicas}) and increased CPU request from {current_cpu_request} to {new_cpu_request}, "
                           f"memory request from {current_memory_request} to {new_memory_request}"
                )
                
            elif should_scale_horizontally:
                # Scale horizontally (add replicas)
                self.logger.info(f"Scaling {namespace}/{pod_name} horizontally")
                
                # Calculate number of new replicas based on CPU usage
                # More aggressive scaling for higher usage
                if cpu_usage > 150:  # Extreme load
                    scale_factor = 3.0
                elif cpu_usage > 120:  # Very high load
                    scale_factor = 2.0
                else:  # High load
                    scale_factor = 1.5
                    
                # Apply scaling with bounds
                new_replicas = min(int(current_replicas * scale_factor) + 1, 10)  # Scale based on load, max 10
                deployment.spec.replicas = new_replicas
                
                # Apply the update
                self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment
                )
                
                self.logger.info(f"Updated {namespace}/{pod_name}: Added replicas (now {new_replicas})")
                
                # Create HPA for future automatic scaling with predicted metrics
                try:
                    # Set the target utilization lower if we're seeing rapid growth
                    target_utilization = 70
                    if pod_id in self.pod_metrics_history and len(self.pod_metrics_history[pod_id]) >= 3:
                        try:
                            # Check if we have a steep upward trend
                            cpu_history = [m.get("cpu_usage", 0) for m in self.pod_metrics_history[pod_id]]
                            x = np.arange(len(cpu_history))
                            cpu_slope, _ = np.polyfit(x, cpu_history, 1)
                            
                            # If CPU usage is growing rapidly, reduce target utilization
                            if cpu_slope > 5:  # More than 5% increase per period
                                target_utilization = max(50, 70 - int(cpu_slope))
                                self.logger.info(f"Setting lower HPA target utilization ({target_utilization}%) due to rapid growth")
                        except Exception as e:
                            self.logger.warning(f"Error calculating CPU trend: {e}")
                    
                    hpa_body = {
                        "apiVersion": "autoscaling/v2",
                        "kind": "HorizontalPodAutoscaler",
                        "metadata": {
                            "name": deployment_name,
                            "namespace": namespace
                        },
                        "spec": {
                            "scaleTargetRef": {
                                "apiVersion": "apps/v1",
                                "kind": "Deployment",
                                "name": deployment_name
                            },
                            "minReplicas": max(1, new_replicas - 1),
                            "maxReplicas": 20,
                            "metrics": [
                                {
                                    "type": "Resource",
                                    "resource": {
                                        "name": "cpu",
                                        "target": {
                                            "type": "Utilization",
                                            "averageUtilization": target_utilization
                                        }
                                    }
                                }
                            ]
                        }
                    }
                    
                    # Try to create the HPA
                    try:
                        self.k8s_api.create_namespaced_horizontal_pod_autoscaler(namespace, hpa_body)
                        self.logger.info(f"Created HPA for {namespace}/{pod_name} with target utilization {target_utilization}%")
                    except Exception as e:
                        self.logger.warning(f"Could not create HPA: {e}")
                except Exception as e:
                    self.logger.warning(f"Error creating HPA: {e}")
                
                # Record the remediation
                self._record_remediation(
                    pod_id=f"{namespace}/{pod_name}",
                    action="scaled_horizontally",
                    success=True,
                    details=f"Added replicas (now {new_replicas}) based on current and predicted load"
                )
                
            elif should_scale_vertically:
                # Scale vertically (increase resources)
                self.logger.info(f"Scaling {namespace}/{pod_name} vertically")
                
                # Calculate scale factors based on current and predicted usage
                cpu_scale_factor = 1.0 + (max(0, (cpu_usage - 80)) / 100)
                memory_scale_factor = 1.0 + (max(0, (memory_usage - 80)) / 100)
                
                # Apply scaling with bounds
                new_cpu_request = f"{int(min(current_cpu_request_val * cpu_scale_factor, current_cpu_request_val * 2.0))}m"
                new_memory_request = f"{int(min(current_memory_request_val * memory_scale_factor, current_memory_request_val * 2.0))}Mi"
                new_cpu_limit = f"{int(min(current_cpu_limit_val * cpu_scale_factor, current_cpu_limit_val * 2.0))}m"
                new_memory_limit = f"{int(min(current_memory_limit_val * memory_scale_factor, current_memory_limit_val * 2.0))}Mi"
                
                # Update resources
                deployment.spec.template.spec.containers[0].resources.requests = {
                    'cpu': new_cpu_request,
                    'memory': new_memory_request
                }
                deployment.spec.template.spec.containers[0].resources.limits = {
                    'cpu': new_cpu_limit,
                    'memory': new_memory_limit
                }
                
                # Apply the update
                self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment
                )
                
                self.logger.info(f"Updated resource requests/limits for {namespace}/{pod_name}")
                self.logger.info(f"New CPU request: {new_cpu_request}, limit: {new_cpu_limit}")
                self.logger.info(f"New memory request: {new_memory_request}, limit: {new_memory_limit}")
                
                # Record the remediation
                self._record_remediation(
                    pod_id=f"{namespace}/{pod_name}",
                    action="scaled_vertically",
                    success=True,
                    details=f"Increased CPU request from {current_cpu_request} to {new_cpu_request}, "
                           f"memory request from {current_memory_request} to {new_memory_request} "
                           f"based on current and predicted resource usage"
                )
            
            else:
                # If we don't need to scale, check if we should create an HPA for future automatic scaling
                if not hpa_exists and (cpu_usage > 60 or memory_usage > 60):
                    self.logger.info(f"Creating HPA for {namespace}/{pod_name} for future automatic scaling")
                    
                    try:
                        hpa_body = {
                            "apiVersion": "autoscaling/v2",
                            "kind": "HorizontalPodAutoscaler",
                            "metadata": {
                                "name": deployment_name,
                                "namespace": namespace
                            },
                            "spec": {
                                "scaleTargetRef": {
                                    "apiVersion": "apps/v1",
                                    "kind": "Deployment",
                                    "name": deployment_name
                                },
                                "minReplicas": max(1, current_replicas),
                                "maxReplicas": 20,
                                "metrics": [
                                    {
                                        "type": "Resource",
                                        "resource": {
                                            "name": "cpu",
                                            "target": {
                                                "type": "Utilization",
                                                "averageUtilization": 70
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                        
                        # Try to create the HPA
                        try:
                            self.k8s_api.create_namespaced_horizontal_pod_autoscaler(namespace, hpa_body)
                            self.logger.info(f"Created HPA for {namespace}/{pod_name}")
                            
                            # Record the remediation
                            self._record_remediation(
                                pod_id=f"{namespace}/{pod_name}",
                                action="created_hpa",
                                success=True,
                                details=f"Created HPA for automatic scaling based on CPU utilization"
                            )
                        except Exception as e:
                            self.logger.warning(f"Could not create HPA: {e}")
                    except Exception as e:
                        self.logger.warning(f"Error creating HPA: {e}")
                
                # No scaling needed
                self.logger.info(f"No scaling needed for {namespace}/{pod_name}")
                return True
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error handling resource issue for {namespace}/{pod_name}: {str(e)}")
            self._record_remediation(
                pod_id=f"{namespace}/{pod_name}",
                action="scaled_resources",
                success=False,
                details=f"Error: {str(e)}"
            )
            return False

    def _handle_core_component_crash_loop(self, pod_name: str, namespace: str, container_id: str) -> bool:
        """Handle crash loops for core components."""
        self.logger.info(f"Handling crash loop for core component {namespace}/{pod_name}")
        
        # Get the pod
        pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
        
        # Check if the pod is using host network mode
        if pod.spec.host_network:
            self.logger.info(f"Core component {namespace}/{pod_name} is using host network mode")
        else:
            self.logger.warning(f"Core component {namespace}/{pod_name} is not using host network mode")
            self.logger.info("Recommended action: Enable host network mode")
            self.logger.info("Command: kubectl patch deployment {pod_name} -n {namespace} -p '{\"spec\":{\"template\":{\"spec\":{\"hostNetwork\":true}}}}'")
        
        # For etcd, check if it's a disk space issue
        if "etcd" in pod_name:
            self.logger.warning("Detected potential disk space issue with etcd")
            
            # Check if etcd has a persistent volume
            if pod.spec.volumes:
                for volume in pod.spec.volumes:
                    if volume.persistent_volume_claim:
                        pvc_name = volume.persistent_volume_claim.claim_name
                        try:
                            pvc = self.k8s_api.read_namespaced_persistent_volume_claim(pvc_name, namespace)
                            self.logger.info(f"etcd PVC {pvc_name} status: {pvc.status.phase}")
                            
                            # Check if the PVC is bound
                            if pvc.status.phase != "Bound":
                                self.logger.error(f"etcd PVC {pvc_name} is not bound")
                                self.logger.info("Recommended action: Check the PVC and PV status")
                                self.logger.info("Command: kubectl get pvc -n kube-system")
                                self.logger.info("Command: kubectl get pv")
                                return False
                            
                            # Check if the PVC has enough storage
                            if pvc.spec.resources.requests:
                                storage_request = pvc.spec.resources.requests.get("storage")
                                if storage_request:
                                    self.logger.info(f"etcd PVC {pvc_name} storage request: {storage_request}")
                                    
                                    # Check if the storage request is less than 10Gi
                                    if self._parse_storage(storage_request) < 10 * 1024 * 1024 * 1024:  # 10Gi in bytes
                                        self.logger.error(f"etcd PVC {pvc_name} storage request is less than 10Gi")
                                        self.logger.info("Recommended action: Increase the PVC storage request")
                                        self.logger.info("Command: kubectl patch pvc {pvc_name} -n kube-system -p '{\"spec\":{\"resources\":{\"requests\":{\"storage\":\"10Gi\"}}}}'")
                                        return False
                        except Exception as e:
                            self.logger.error(f"Error getting PVC {pvc_name}: {e}")
                            return False
                    elif volume.host_path:
                        self.logger.info(f"etcd is using host path volume: {volume.host_path.path}")
                        
                        # Check if the host path has the correct permissions
                        # This would require access to the host, which we don't have
                        self.logger.warning("Cannot check host path volume permissions")
                        self.logger.info("Recommended action: Check disk space on the node")
                        self.logger.info("Command: df -h")
                        return False
            else:
                self.logger.warning("etcd does not have any volumes")
                self.logger.info("Recommended action: Add a persistent volume to etcd")
                self.logger.info("Command: kubectl edit deployment etcd-minikube -n kube-system")
                return False
        
        # For kube-apiserver, check if it's a metrics API issue
        if "kube-apiserver" in pod_name:
            self.logger.warning("Detected potential metrics API issue with kube-apiserver")
            
            # Check if the metrics-server is running
            try:
                metrics_server = self.k8s_api.read_namespaced_deployment("metrics-server", "kube-system")
                if metrics_server.status.ready_replicas != metrics_server.spec.replicas:
                    self.logger.error("metrics-server is not running")
                    self.logger.info("Recommended action: Check the metrics-server deployment")
                    self.logger.info("Command: kubectl get deployment metrics-server -n kube-system")
                    self.logger.info("Command: kubectl describe deployment metrics-server -n kube-system")
                    return False
            except Exception as e:
                self.logger.error(f"Error getting metrics-server deployment: {e}")
                self.logger.info("Recommended action: Check if the metrics-server deployment exists")
                self.logger.info("Command: kubectl get deployment metrics-server -n kube-system")
                return False
            
            # Check if the metrics API is registered
            try:
                api_services = self.k8s_api.list_api_service()
                metrics_api_registered = False
                for api_service in api_services.items:
                    if api_service.metadata.name == "v1beta1.metrics.k8s.io":
                        metrics_api_registered = True
                        break
                
                if not metrics_api_registered:
                    self.logger.error("metrics API is not registered")
                    self.logger.info("Recommended action: Check the metrics API registration")
                    self.logger.info("Command: kubectl get apiservice v1beta1.metrics.k8s.io")
                    return False
            except Exception as e:
                self.logger.error(f"Error checking metrics API registration: {e}")
                return False
        
        # For kube-controller-manager, check if it's a permission issue
        if "kube-controller-manager" in pod_name:
            self.logger.warning("Detected potential permission issue with kube-controller-manager")
            
            # Check if the horizontal-pod-autoscaler service account exists
            try:
                service_account = self.k8s_api.read_namespaced_service_account("horizontal-pod-autoscaler", "kube-system")
                self.logger.info("horizontal-pod-autoscaler service account exists")
            except Exception as e:
                self.logger.error("horizontal-pod-autoscaler service account does not exist")
                self.logger.info("Recommended action: Create the horizontal-pod-autoscaler service account")
                self.logger.info("Command: kubectl create serviceaccount horizontal-pod-autoscaler -n kube-system")
                return False
            
            # Check if the horizontal-pod-autoscaler role binding exists
            try:
                role_binding = self.k8s_api.read_namespaced_role_binding("horizontal-pod-autoscaler", "kube-system")
                self.logger.info("horizontal-pod-autoscaler role binding exists")
            except Exception as e:
                self.logger.error("horizontal-pod-autoscaler role binding does not exist")
                self.logger.info("Recommended action: Create the horizontal-pod-autoscaler role binding")
                self.logger.info("Command: kubectl create rolebinding horizontal-pod-autoscaler --clusterrole=system:horizontal-pod-autoscaler --serviceaccount=kube-system:horizontal-pod-autoscaler -n kube-system")
                return False
        
        # For kube-scheduler, check if it's an authentication issue
        if "kube-scheduler" in pod_name:
            self.logger.warning("Detected potential authentication issue with kube-scheduler")
            
            # Check if the extension-apiserver-authentication configmap exists
            try:
                configmap = self.k8s_api.read_namespaced_config_map("extension-apiserver-authentication", "kube-system")
                self.logger.info("extension-apiserver-authentication configmap exists")
            except Exception as e:
                self.logger.error("extension-apiserver-authentication configmap does not exist")
                self.logger.info("Recommended action: Create the extension-apiserver-authentication configmap")
                self.logger.info("Command: kubectl create configmap extension-apiserver-authentication --from-literal=client-ca-file=/etc/kubernetes/pki/ca.crt --from-literal=requestheader-client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt -n kube-system")
                return False
        
        # If we've made it this far, we can try to remediate the crash loop
        self.logger.info(f"Attempting to remediate crash loop for core component {namespace}/{pod_name}")
        
        # Delete the pod to force a restart
        try:
            self.k8s_api.delete_namespaced_pod(pod_name, namespace)
            self.logger.info(f"Deleted pod {namespace}/{pod_name} to force a restart")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting pod {namespace}/{pod_name}: {e}")
            return False 

    def _get_deployment_name(self, pod_name: str, namespace: str) -> Optional[str]:
        """Get the deployment name for a pod."""
        try:
            # Get the pod
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Check owner references
            if pod.metadata.owner_references:
                for ref in pod.metadata.owner_references:
                    if ref.kind == 'ReplicaSet':
                        # Get the ReplicaSet
                        rs = self.k8s_apps_api.read_namespaced_replica_set(ref.name, namespace)
                        # Check ReplicaSet owner references
                        if rs.metadata.owner_references:
                            for rs_ref in rs.metadata.owner_references:
                                if rs_ref.kind == 'Deployment':
                                    return rs_ref.name
            
            # If no deployment found, try to extract from pod name
            # This is a fallback for pods that might not have proper owner references
            parts = pod_name.split('-')
            if len(parts) > 1:
                # Remove the last part (usually a random string)
                return '-'.join(parts[:-1])
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting deployment name for pod {pod_name}: {str(e)}")
            return None 

    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU string to float value."""
        try:
            if cpu_str.endswith('m'):
                return float(cpu_str[:-1]) / 1000
            elif cpu_str.endswith('n'):
                return float(cpu_str[:-1]) / 1000000000
            else:
                return float(cpu_str)
        except (ValueError, TypeError):
            return 0.0

    def _check_api_service(self, name: str) -> bool:
        """Check if an API service exists and is available."""
        try:
            # Use the API registration API instead of list_api_service
            api_registration = self.k8s_api.get_api_resources()
            for api in api_registration:
                if api.name == name:
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking API service {name}: {str(e)}")
            return False 

    def _parse_storage(self, storage_str: str) -> int:
        """Parse storage string to bytes."""
        try:
            if not storage_str:
                return 0
            
            if storage_str.endswith('Ki'):
                return int(float(storage_str[:-2]) * 1024)
            elif storage_str.endswith('Mi'):
                return int(float(storage_str[:-2]) * 1024 * 1024)
            elif storage_str.endswith('Gi'):
                return int(float(storage_str[:-2]) * 1024 * 1024 * 1024)
            elif storage_str.endswith('Ti'):
                return int(float(storage_str[:-2]) * 1024 * 1024 * 1024 * 1024)
            elif storage_str.endswith('Pi'):
                return int(float(storage_str[:-2]) * 1024 * 1024 * 1024 * 1024 * 1024)
            elif storage_str.endswith('Ei'):
                return int(float(storage_str[:-2]) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024)
            # Handle lowercase suffixes
            elif storage_str.endswith('ki'):
                return int(float(storage_str[:-2]) * 1024)
            elif storage_str.endswith('mi'):
                return int(float(storage_str[:-2]) * 1024 * 1024)
            elif storage_str.endswith('gi'):
                return int(float(storage_str[:-2]) * 1024 * 1024 * 1024)
            elif storage_str.endswith('ti'):
                return int(float(storage_str[:-2]) * 1024 * 1024 * 1024 * 1024)
            elif storage_str.endswith('pi'):
                return int(float(storage_str[:-2]) * 1024 * 1024 * 1024 * 1024 * 1024)
            elif storage_str.endswith('ei'):
                return int(float(storage_str[:-2]) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024)
            # Handle plain bytes
            elif storage_str.isdigit():
                return int(storage_str)
            else:
                # Try to convert directly
                return int(float(storage_str))
        except (ValueError, TypeError):
            return 0 

    def _handle_crash_loop(self, pod_name: str, namespace: str, container_id: str) -> bool:
        """Handle a pod that is in a crash loop state."""
        try:
            # Get the pod details
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Get the container status
            container_status = next(
                (status for status in pod.status.container_statuses 
                 if status.container_id and status.container_id.split('//')[-1] == container_id),
                None
            )
            
            if not container_status:
                self.logger.error(f"Could not find container status for {pod_name}")
                return False
                
            # Get the last state and its details
            last_state = container_status.last_state
            if last_state.terminated:
                exit_code = last_state.terminated.exit_code
                reason = last_state.terminated.reason
                message = last_state.terminated.message
                
                self.logger.info(f"Pod {pod_name} last exit: code={exit_code}, reason={reason}, message={message}")
                
                # Check if it's a configuration issue
                if exit_code == 127:  # Command not found
                    self.logger.info(f"Pod {pod_name} is failing due to invalid command/configuration")
                    return self._handle_configuration_error(pod_name, namespace)
                
                # Check if it's an OOM kill
                elif exit_code == 137 or reason == "OOMKilled":
                    self.logger.info(f"Pod {pod_name} was killed due to OOM")
                    return self._handle_resource_exhaustion(pod_name, namespace, container_id)
                
                # Check if it's a permission issue
                elif exit_code == 126:  # Permission denied
                    self.logger.info(f"Pod {pod_name} is failing due to permission issues")
                    return self._handle_permission_error(pod_name, namespace)
            
            # Get the deployment that owns this pod
            owner_reference = next(
                (ref for ref in pod.metadata.owner_references if ref.kind == "ReplicaSet"),
                None
            )
            
            if owner_reference:
                # Get the ReplicaSet to find the Deployment
                rs = self.k8s_apps_api.read_namespaced_replica_set(
                    owner_reference.name,
                    namespace
                )
                deployment_name = rs.metadata.owner_references[0].name
                
                # Scale down and back up the deployment
                self.logger.info(f"Attempting to restart deployment {deployment_name}")
                try:
                    # Scale down to 0
                    self.k8s_apps_api.patch_namespaced_deployment_scale(
                        deployment_name,
                        namespace,
                        {"spec": {"replicas": 0}}
                    )
                    
                    time.sleep(5)  # Wait for scale down
                    
                    # Scale back up to 1
                    self.k8s_apps_api.patch_namespaced_deployment_scale(
                        deployment_name,
                        namespace,
                        {"spec": {"replicas": 1}}
                    )
                    
                    self.logger.info(f"Successfully restarted deployment {deployment_name}")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error restarting deployment {deployment_name}: {e}")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error handling crash loop for {pod_name}: {e}")
            return False

    def _handle_configuration_error(self, pod_name: str, namespace: str) -> bool:
        """Handle configuration errors in pods."""
        try:
            # Get the pod's logs to analyze the error
            logs = self.k8s_api.read_namespaced_pod_log(
                pod_name,
                namespace,
                container='crash-test',
                tail_lines=50
            )
            
            self.logger.info(f"Configuration error in {pod_name}. Logs:\n{logs}")
            
            # For now, just log the error. In a real implementation,
            # you might want to analyze the logs and make configuration adjustments
            return False
            
        except Exception as e:
            self.logger.error(f"Error handling configuration error for {pod_name}: {e}")
            return False

    def _handle_permission_error(self, pod_name: str, namespace: str) -> bool:
        """Handle permission-related errors in pods."""
        try:
            # Get the pod
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Check if pod has a service account
            if not pod.spec.service_account_name:
                self.logger.info(f"Pod {pod_name} has no service account. This might be the cause of permission issues.")
            else:
                self.logger.info(f"Pod {pod_name} is using service account: {pod.spec.service_account_name}")
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error handling permission error for {pod_name}: {e}")
            return False