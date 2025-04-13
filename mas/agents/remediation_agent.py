from .base_agent import BaseAgent
from langchain.agents import Tool
from typing import List, Dict, Any
import kubernetes as k8s
import os

class RemediationAgent(BaseAgent):
    """Agent responsible for remediating issues in the Kubernetes cluster"""
    
    def __init__(self, llm):
        """Initialize the remediation agent"""
        super().__init__(llm)
        self.k8s_client = k8s.client.CoreV1Api()
        self.apps_client = k8s.client.AppsV1Api()
        self.rbac_client = k8s.client.RbacAuthorizationV1Api()
    
    def get_tools(self) -> List[Tool]:
        """Get the remediation tools available to this agent"""
        return [
            Tool(
                name="restart_pod",
                func=self._restart_pod,
                description="Restart a pod by deleting it (Kubernetes will recreate it)"
            ),
            Tool(
                name="scale_deployment",
                func=self._scale_deployment,
                description="Scale a deployment to a specific number of replicas"
            ),
            Tool(
                name="update_resource_limits",
                func=self._update_resource_limits,
                description="Update resource limits for a deployment"
            ),
            Tool(
                name="apply_configmap",
                func=self._apply_configmap,
                description="Apply a ConfigMap to update configuration"
            ),
            Tool(
                name="create_role_binding",
                func=self._create_role_binding,
                description="Create a role binding to grant permissions to a service account"
            ),
            Tool(
                name="create_cluster_role_binding",
                func=self._create_cluster_role_binding,
                description="Create a cluster role binding to grant cluster-wide permissions"
            ),
            Tool(
                name="check_service_account",
                func=self._check_service_account,
                description="Check if a service account exists and has the necessary permissions"
            ),
            Tool(
                name="fix_metrics_server",
                func=self._fix_metrics_server,
                description="Fix issues with the metrics-server deployment"
            )
        ]
    
    def _restart_pod(self, pod_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Restart a pod by deleting it"""
        try:
            self.k8s_client.delete_namespaced_pod(pod_name, namespace)
            return {
                'success': True,
                'action': 'pod_restart',
                'pod_name': pod_name,
                'namespace': namespace,
                'message': f"Pod {pod_name} in namespace {namespace} has been restarted"
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _scale_deployment(self, deployment_name: str, replicas: int, namespace: str = "default") -> Dict[str, Any]:
        """Scale a deployment to a specific number of replicas"""
        try:
            # Get the current deployment
            deployment = self.apps_client.read_namespaced_deployment(deployment_name, namespace)
            
            # Update the replicas
            deployment.spec.replicas = replicas
            
            # Apply the update
            self.apps_client.patch_namespaced_deployment(deployment_name, namespace, deployment)
            
            return {
                'success': True,
                'action': 'scale_deployment',
                'deployment_name': deployment_name,
                'namespace': namespace,
                'replicas': replicas,
                'message': f"Deployment {deployment_name} in namespace {namespace} scaled to {replicas} replicas"
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _update_resource_limits(self, deployment_name: str, namespace: str = "default", 
                               cpu_limit: str = None, memory_limit: str = None) -> Dict[str, Any]:
        """Update resource limits for a deployment"""
        try:
            # Get the current deployment
            deployment = self.apps_client.read_namespaced_deployment(deployment_name, namespace)
            
            # Update resource limits for all containers
            for container in deployment.spec.template.spec.containers:
                if not container.resources:
                    container.resources = k8s.client.V1ResourceRequirements()
                if not container.resources.limits:
                    container.resources.limits = {}
                
                if cpu_limit:
                    container.resources.limits['cpu'] = cpu_limit
                if memory_limit:
                    container.resources.limits['memory'] = memory_limit
            
            # Apply the update
            self.apps_client.patch_namespaced_deployment(deployment_name, namespace, deployment)
            
            return {
                'success': True,
                'action': 'update_resource_limits',
                'deployment_name': deployment_name,
                'namespace': namespace,
                'cpu_limit': cpu_limit,
                'memory_limit': memory_limit,
                'message': f"Resource limits updated for deployment {deployment_name} in namespace {namespace}"
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _apply_configmap(self, configmap_name: str, namespace: str = "default", 
                        data: Dict[str, str] = None) -> Dict[str, Any]:
        """Apply a ConfigMap to update configuration"""
        try:
            # Create or update the ConfigMap
            configmap = k8s.client.V1ConfigMap(
                metadata=k8s.client.V1ObjectMeta(name=configmap_name),
                data=data or {}
            )
            
            try:
                # Try to update existing ConfigMap
                self.k8s_client.patch_namespaced_config_map(configmap_name, namespace, configmap)
                action = "updated"
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    # ConfigMap doesn't exist, create it
                    self.k8s_client.create_namespaced_config_map(namespace, configmap)
                    action = "created"
                else:
                    raise
            
            return {
                'success': True,
                'action': 'apply_configmap',
                'configmap_name': configmap_name,
                'namespace': namespace,
                'message': f"ConfigMap {configmap_name} {action} in namespace {namespace}"
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _create_role_binding(self, name: str, role: str, service_account: str, namespace: str) -> Dict[str, Any]:
        """Create a role binding to grant permissions to a service account"""
        try:
            # Parse service account (format: namespace:name)
            sa_namespace, sa_name = service_account.split(':')
            
            # Create the role binding
            role_binding = k8s.client.V1RoleBinding(
                metadata=k8s.client.V1ObjectMeta(name=name),
                role_ref=k8s.client.V1RoleRef(
                    api_group="rbac.authorization.k8s.io",
                    kind="Role",
                    name=role
                ),
                subjects=[
                    k8s.client.V1Subject(
                        kind="ServiceAccount",
                        name=sa_name,
                        namespace=sa_namespace
                    )
                ]
            )
            
            try:
                # Try to update existing role binding
                self.rbac_client.patch_namespaced_role_binding(name, namespace, role_binding)
                action = "updated"
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    # Role binding doesn't exist, create it
                    self.rbac_client.create_namespaced_role_binding(namespace, role_binding)
                    action = "created"
                else:
                    raise
            
            return {
                'success': True,
                'action': 'create_role_binding',
                'name': name,
                'role': role,
                'service_account': service_account,
                'namespace': namespace,
                'message': f"Role binding {name} {action} in namespace {namespace}"
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _create_cluster_role_binding(self, name: str, role: str, service_account: str) -> Dict[str, Any]:
        """Create a cluster role binding to grant cluster-wide permissions"""
        try:
            # Parse service account (format: namespace:name)
            sa_namespace, sa_name = service_account.split(':')
            
            # Create the cluster role binding
            cluster_role_binding = k8s.client.V1ClusterRoleBinding(
                metadata=k8s.client.V1ObjectMeta(name=name),
                role_ref=k8s.client.V1RoleRef(
                    api_group="rbac.authorization.k8s.io",
                    kind="ClusterRole",
                    name=role
                ),
                subjects=[
                    k8s.client.V1Subject(
                        kind="ServiceAccount",
                        name=sa_name,
                        namespace=sa_namespace
                    )
                ]
            )
            
            try:
                # Try to update existing cluster role binding
                self.rbac_client.patch_cluster_role_binding(name, cluster_role_binding)
                action = "updated"
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    # Cluster role binding doesn't exist, create it
                    self.rbac_client.create_cluster_role_binding(cluster_role_binding)
                    action = "created"
                else:
                    raise
            
            return {
                'success': True,
                'action': 'create_cluster_role_binding',
                'name': name,
                'role': role,
                'service_account': service_account,
                'message': f"Cluster role binding {name} {action}"
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _check_service_account(self, name: str, namespace: str) -> Dict[str, Any]:
        """Check if a service account exists and has the necessary permissions"""
        try:
            # Check if the service account exists
            service_account = self.k8s_client.read_namespaced_service_account(name, namespace)
            
            # Get role bindings for the service account
            role_bindings = self.rbac_client.list_namespaced_role_binding(namespace)
            service_account_role_bindings = [
                binding for binding in role_bindings.items
                if any(
                    subject.kind == "ServiceAccount" and 
                    subject.name == name and 
                    subject.namespace == namespace
                    for subject in binding.subjects
                )
            ]
            
            # Get cluster role bindings for the service account
            cluster_role_bindings = self.rbac_client.list_cluster_role_binding()
            service_account_cluster_role_bindings = [
                binding for binding in cluster_role_bindings.items
                if any(
                    subject.kind == "ServiceAccount" and 
                    subject.name == name and 
                    subject.namespace == namespace
                    for subject in binding.subjects
                )
            ]
            
            return {
                'success': True,
                'exists': True,
                'name': name,
                'namespace': namespace,
                'role_bindings': [
                    {
                        'name': binding.metadata.name,
                        'role': binding.role_ref.name,
                        'role_kind': binding.role_ref.kind
                    }
                    for binding in service_account_role_bindings
                ],
                'cluster_role_bindings': [
                    {
                        'name': binding.metadata.name,
                        'role': binding.role_ref.name,
                        'role_kind': binding.role_ref.kind
                    }
                    for binding in service_account_cluster_role_bindings
                ],
                'message': f"Service account {name} in namespace {namespace} exists with {len(service_account_role_bindings)} role bindings and {len(service_account_cluster_role_bindings)} cluster role bindings"
            }
        except k8s.client.rest.ApiException as e:
            if e.status == 404:
                return {
                    'success': True,
                    'exists': False,
                    'name': name,
                    'namespace': namespace,
                    'message': f"Service account {name} in namespace {namespace} does not exist"
                }
            else:
                return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)
    
    def _fix_metrics_server(self) -> Dict[str, Any]:
        """Fix issues with the metrics-server deployment"""
        try:
            # Check if metrics-server deployment exists
            try:
                metrics_server = self.apps_client.read_namespaced_deployment("metrics-server", "kube-system")
                exists = True
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    exists = False
                else:
                    raise
            
            if not exists:
                # Create metrics-server deployment
                # This is a simplified example - in a real system, you would have a more complete deployment
                deployment = k8s.client.V1Deployment(
                    metadata=k8s.client.V1ObjectMeta(name="metrics-server"),
                    spec=k8s.client.V1DeploymentSpec(
                        replicas=1,
                        selector=k8s.client.V1LabelSelector(
                            match_labels={"k8s-app": "metrics-server"}
                        ),
                        template=k8s.client.V1PodTemplateSpec(
                            metadata=k8s.client.V1ObjectMeta(
                                labels={"k8s-app": "metrics-server"}
                            ),
                            spec=k8s.client.V1PodSpec(
                                service_account_name="metrics-server",
                                containers=[
                                    k8s.client.V1Container(
                                        name="metrics-server",
                                        image="registry.k8s.io/metrics-server/metrics-server:v0.6.3",
                                        args=["--cert-dir=/tmp", "--secure-port=4443", "--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname", "--kubelet-use-node-status-port", "--metric-resolution=15s", "--kubelet-insecure-tls"],
                                        ports=[
                                            k8s.client.V1ContainerPort(
                                                container_port=4443,
                                                protocol="TCP"
                                            )
                                        ],
                                        resources=k8s.client.V1ResourceRequirements(
                                            requests={
                                                "cpu": "100m",
                                                "memory": "200Mi"
                                            },
                                            limits={
                                                "cpu": "200m",
                                                "memory": "400Mi"
                                            }
                                        )
                                    )
                                ]
                            )
                        )
                    )
                )
                
                self.apps_client.create_namespaced_deployment("kube-system", deployment)
                
                # Create service account if it doesn't exist
                try:
                    self.k8s_client.read_namespaced_service_account("metrics-server", "kube-system")
                except k8s.client.rest.ApiException as e:
                    if e.status == 404:
                        service_account = k8s.client.V1ServiceAccount(
                            metadata=k8s.client.V1ObjectMeta(name="metrics-server")
                        )
                        self.k8s_client.create_namespaced_service_account("kube-system", service_account)
                
                # Create cluster role binding if it doesn't exist
                try:
                    self.rbac_client.read_cluster_role_binding("system:metrics-server")
                except k8s.client.rest.ApiException as e:
                    if e.status == 404:
                        cluster_role_binding = k8s.client.V1ClusterRoleBinding(
                            metadata=k8s.client.V1ObjectMeta(name="system:metrics-server"),
                            role_ref=k8s.client.V1RoleRef(
                                api_group="rbac.authorization.k8s.io",
                                kind="ClusterRole",
                                name="system:metrics-server"
                            ),
                            subjects=[
                                k8s.client.V1Subject(
                                    kind="ServiceAccount",
                                    name="metrics-server",
                                    namespace="kube-system"
                                )
                            ]
                        )
                        self.rbac_client.create_cluster_role_binding(cluster_role_binding)
                
                return {
                    'success': True,
                    'action': 'create_metrics_server',
                    'message': "Created metrics-server deployment, service account, and cluster role binding"
                }
            else:
                # Update existing metrics-server deployment
                # This is a simplified example - in a real system, you would have a more complete update
                metrics_server.spec.template.spec.containers[0].args = [
                    "--cert-dir=/tmp", 
                    "--secure-port=4443", 
                    "--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname", 
                    "--kubelet-use-node-status-port", 
                    "--metric-resolution=15s", 
                    "--kubelet-insecure-tls"
                ]
                
                self.apps_client.patch_namespaced_deployment("metrics-server", "kube-system", metrics_server)
                
                return {
                    'success': True,
                    'action': 'update_metrics_server',
                    'message': "Updated metrics-server deployment"
                }
        except Exception as e:
            return self.handle_error(e)
    
    def process_result(self, result: str) -> Dict[str, Any]:
        """Process the result from remediation actions"""
        # TODO: Implement result processing logic
        return {
            'status': 'processed',
            'result': result
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for remediation"""
        required_fields = ['action', 'target']
        return all(field in input_data for field in required_fields)
    
    def format_output(self, output_data: Dict[str, Any]) -> str:
        """Format the remediation output data"""
        # TODO: Implement output formatting logic
        return str(output_data) 