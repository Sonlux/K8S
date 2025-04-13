#!/usr/bin/env python3

import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from kubernetes import client
from .logging_observability import LoggingObservability

class RemediationSystem:
    """Remediation system for Kubernetes resources."""
    
    def __init__(self, k8s_api: client.CoreV1Api,
                 apps_api: client.AppsV1Api,
                 logging: LoggingObservability,
                 cooldown_period=300,
                 max_attempts=3):
        """Initialize the remediation system."""
        self.k8s_api = k8s_api
        self.apps_api = apps_api
        self.logging = logging
        self.cooldown_period = cooldown_period
        self.max_attempts = max_attempts
        
        # Track remediation history for cooldown
        self.remediation_history = {}
        
        # Track state snapshots for rollback
        self.state_snapshots = {}
        
        # Define remediation strategies
        self.strategies = {
            "oom": {
                "action": self._handle_oom,
                "description": "Remediate out-of-memory issues by increasing memory limits"
            },
            "crash_loop": {
                "action": self._handle_crash_loop,
                "description": "Remediate crash loops by analyzing logs and restarting pods"
            },
            "image_pull": {
                "action": self._handle_image_pull,
                "description": "Remediate image pull issues by checking image registry access"
            },
            "network": {
                "action": self._handle_network,
                "description": "Remediate network issues by checking DNS and network policies"
            },
            "resource": {
                "action": self._handle_resource,
                "description": "Remediate resource issues by scaling CPU and memory"
            }
        }
    
    def remediate(self, error: Dict[str, Any]) -> bool:
        """Remediate an error."""
        error_type = error["type"]
        namespace = error["namespace"]
        pod_name = error["pod"]
        
        # Check if remediation is needed
        if not self._should_remediate(error_type, namespace, pod_name):
            return False
        
        try:
            # Get remediation strategy
            strategy = self.strategies.get(error_type)
            if not strategy:
                self.logging.log_event(
                    "remediation_skipped",
                    f"No remediation strategy for error type: {error_type}",
                    namespace,
                    pod_name
                )
                return False
            
            # Take a snapshot before remediation
            deployment_name = self._get_deployment_from_pod(namespace, pod_name)
            if deployment_name:
                self._snapshot_resource_state(namespace, 'deployment', deployment_name)
            self._snapshot_resource_state(namespace, 'pod', pod_name)
            
            # Execute remediation action
            start_time = time.time()
            success = strategy["action"](error)
            duration = time.time() - start_time
            
            # Update history
            self._update_history(error_type, namespace, pod_name, success)
            
            # Log result
            self.logging.record_remediation_duration(
                error_type,
                duration,
                {"namespace": namespace, "pod": pod_name}
            )
            
            if success:
                self.logging.log_event(
                    "remediation_succeeded",
                    f"Successfully remediated {error_type} error",
                    namespace,
                    pod_name
                )
            else:
                self.logging.log_event(
                    "remediation_failed",
                    f"Failed to remediate {error_type} error",
                    namespace,
                    pod_name
                )
                
                # Attempt rollback if remediation failed
                if deployment_name:
                    self.logging.log_event(
                        "rollback_started",
                        f"Starting rollback for failed remediation of {error_type} error",
                        namespace,
                        pod_name
                    )
                    rollback_success = self._rollback_resource(namespace, 'deployment', deployment_name)
                    if rollback_success:
                        self.logging.log_event(
                            "rollback_completed",
                            f"Successfully rolled back {deployment_name} after failed remediation",
                            namespace,
                            pod_name
                        )
                    else:
                        self.logging.log_event(
                            "rollback_failed",
                            f"Failed to roll back {deployment_name} after failed remediation",
                            namespace,
                            pod_name
                        )
            
            return success
        except Exception as e:
            self.logging.log_event(
                "remediation_error",
                f"Error during remediation: {str(e)}",
                namespace,
                pod_name
            )
            return False
    
    def _should_remediate(self, error_type: str, namespace: str,
                         pod_name: str) -> bool:
        """Check if remediation should be attempted."""
        key = f"{namespace}/{pod_name}/{error_type}"
        now = datetime.utcnow()
        
        if key not in self.remediation_history:
            return True
        
        history = self.remediation_history[key]
        strategy = self.strategies.get(error_type)
        
        if not strategy:
            return False
        
        # Check cooldown period
        if history["last_attempt"]:
            cooldown = timedelta(seconds=self.cooldown_period)
            if now - history["last_attempt"] < cooldown:
                return False
        
        # Check max attempts
        if history["attempts"] >= self.max_attempts:
            return False
        
        return True
    
    def _update_history(self, error_type: str, namespace: str,
                       pod_name: str, success: bool):
        """Update remediation history."""
        key = f"{namespace}/{pod_name}/{error_type}"
        now = datetime.utcnow()
        
        if key not in self.remediation_history:
            self.remediation_history[key] = {
                "attempts": 0,
                "successes": 0,
                "last_attempt": None,
                "last_success": None
            }
        
        history = self.remediation_history[key]
        history["attempts"] += 1
        history["last_attempt"] = now
        
        if success:
            history["successes"] += 1
            history["last_success"] = now
    
    def _handle_oom(self, error: Dict[str, Any]) -> bool:
        """Handle out-of-memory errors."""
        namespace = error["namespace"]
        pod_name = error["pod"]
        
        try:
            # Get pod
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Get deployment
            deployment_name = None
            for owner in pod.metadata.owner_references or []:
                if owner.kind == "ReplicaSet":
                    rs = self.apps_api.read_namespaced_replica_set(
                        owner.name, namespace
                    )
                    for rs_owner in rs.metadata.owner_references or []:
                        if rs_owner.kind == "Deployment":
                            deployment_name = rs_owner.name
                            break
            
            if not deployment_name:
                return False
            
            # Get deployment
            deployment = self.apps_api.read_namespaced_deployment(
                deployment_name, namespace
            )
            
            # Calculate new resource limits
            current_limits = deployment.spec.template.spec.containers[0].resources.limits
            if not current_limits:
                current_limits = {}
            
            memory_limit = current_limits.get("memory", "512Mi")
            new_limit = self._increase_memory_limit(memory_limit)
            
            # Update deployment
            deployment.spec.template.spec.containers[0].resources.limits = {
                "memory": new_limit
            }
            
            self.apps_api.patch_namespaced_deployment(
                deployment_name, namespace, deployment
            )
            
            return True
        
        except Exception as e:
            self.logging.log_event(
                "oom_remediation_failed",
                f"Failed to handle OOM: {str(e)}",
                namespace,
                pod_name
            )
            return False
    
    def _handle_crash_loop(self, error: Dict[str, Any]) -> bool:
        """Handle crash loop errors."""
        namespace = error["namespace"]
        pod_name = error["pod"]
        
        try:
            # Delete pod to trigger recreation
            self.k8s_api.delete_namespaced_pod(
                pod_name, namespace,
                grace_period_seconds=30
            )
            
            # Wait for pod to be recreated
            for _ in range(30):  # 5 minutes timeout
                try:
                    pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
                    if pod.status.phase == "Running":
                        return True
                except client.rest.ApiException as e:
                    if e.status == 404:  # Pod not found
                        time.sleep(10)
                        continue
                    raise
                
                time.sleep(10)
            
            return False
        
        except Exception as e:
            self.logging.log_event(
                "crash_loop_remediation_failed",
                f"Failed to handle crash loop: {str(e)}",
                namespace,
                pod_name
            )
            return False
    
    def _handle_image_pull(self, error: Dict[str, Any]) -> bool:
        """Handle image pull errors."""
        namespace = error["namespace"]
        pod_name = error["pod"]
        
        try:
            # Get pod
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Check image pull policy
            for container in pod.spec.containers:
                if container.image_pull_policy != "Always":
                    # Update deployment to force image pull
                    deployment_name = None
                    for owner in pod.metadata.owner_references or []:
                        if owner.kind == "ReplicaSet":
                            rs = self.apps_api.read_namespaced_replica_set(
                                owner.name, namespace
                            )
                            for rs_owner in rs.metadata.owner_references or []:
                                if rs_owner.kind == "Deployment":
                                    deployment_name = rs_owner.name
                                    break
                    
                    if deployment_name:
                        deployment = self.apps_api.read_namespaced_deployment(
                            deployment_name, namespace
                        )
                        
                        # Add timestamp annotation to force update
                        deployment.metadata.annotations = {
                            **(deployment.metadata.annotations or {}),
                            "kubectl.kubernetes.io/restartedAt": datetime.utcnow().isoformat()
                        }
                        
                        self.apps_api.patch_namespaced_deployment(
                            deployment_name, namespace, deployment
                        )
                        
                        return True
            
            return False
        
        except Exception as e:
            self.logging.log_event(
                "image_pull_remediation_failed",
                f"Failed to handle image pull error: {str(e)}",
                namespace,
                pod_name
            )
            return False
    
    def _handle_network(self, error: Dict[str, Any]) -> bool:
        """Handle network errors."""
        namespace = error["namespace"]
        pod_name = error["pod"]
        
        try:
            # Get pod
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Check if using host network
            if not pod.spec.host_network:
                # Update deployment to use host network
                deployment_name = None
                for owner in pod.metadata.owner_references or []:
                    if owner.kind == "ReplicaSet":
                        rs = self.apps_api.read_namespaced_replica_set(
                            owner.name, namespace
                        )
                        for rs_owner in rs.metadata.owner_references or []:
                            if rs_owner.kind == "Deployment":
                                deployment_name = rs_owner.name
                                break
                
                if deployment_name:
                    deployment = self.apps_api.read_namespaced_deployment(
                        deployment_name, namespace
                    )
                    
                    # Enable host network
                    deployment.spec.template.spec.host_network = True
                    
                    self.apps_api.patch_namespaced_deployment(
                        deployment_name, namespace, deployment
                    )
                    
                    return True
            
            return False
        
        except Exception as e:
            self.logging.log_event(
                "network_remediation_failed",
                f"Failed to handle network error: {str(e)}",
                namespace,
                pod_name
            )
            return False
    
    def _handle_resource(self, error: Dict[str, Any]) -> bool:
        """Handle resource errors."""
        namespace = error["namespace"]
        pod_name = error["pod"]
        
        try:
            # Get pod
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Get deployment
            deployment_name = None
            for owner in pod.metadata.owner_references or []:
                if owner.kind == "ReplicaSet":
                    rs = self.apps_api.read_namespaced_replica_set(
                        owner.name, namespace
                    )
                    for rs_owner in rs.metadata.owner_references or []:
                        if rs_owner.kind == "Deployment":
                            deployment_name = rs_owner.name
                            break
            
            if not deployment_name:
                return False
            
            # Get deployment
            deployment = self.apps_api.read_namespaced_deployment(
                deployment_name, namespace
            )
            
            # Update resource requests and limits
            container = deployment.spec.template.spec.containers[0]
            if not container.resources:
                container.resources = client.V1ResourceRequirements()
            
            # Set resource requests
            if not container.resources.requests:
                container.resources.requests = {}
            
            container.resources.requests.update({
                "cpu": "100m",
                "memory": "128Mi"
            })
            
            # Set resource limits
            if not container.resources.limits:
                container.resources.limits = {}
            
            container.resources.limits.update({
                "cpu": "500m",
                "memory": "512Mi"
            })
            
            self.apps_api.patch_namespaced_deployment(
                deployment_name, namespace, deployment
            )
            
            return True
        
        except Exception as e:
            self.logging.log_event(
                "resource_remediation_failed",
                f"Failed to handle resource error: {str(e)}",
                namespace,
                pod_name
            )
            return False
    
    def _increase_memory_limit(self, current_limit: str) -> str:
        """Increase memory limit by 50%."""
        try:
            # Parse current limit
            value = int(current_limit[:-2])
            unit = current_limit[-2:]
            
            # Calculate new limit
            new_value = int(value * 1.5)
            
            # Convert to appropriate unit
            if unit == "Mi":
                if new_value >= 1024:
                    return f"{new_value // 1024}Gi"
                return f"{new_value}Mi"
            elif unit == "Gi":
                return f"{new_value}Gi"
            
            return current_limit
        
        except (ValueError, IndexError):
            return current_limit
    
    def get_remediation_summary(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of remediation actions."""
        summary = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "by_type": {},
            "by_namespace": {}
        }
        
        for key, history in self.remediation_history.items():
            ns, pod, error_type = key.split('/')
            
            if namespace and ns != namespace:
                continue
            
            # Update totals
            summary["total_attempts"] += history["attempts"]
            summary["successful_attempts"] += history["successes"]
            
            # By type
            if error_type not in summary["by_type"]:
                summary["by_type"][error_type] = {
                    "attempts": 0,
                    "successes": 0
                }
            
            summary["by_type"][error_type]["attempts"] += history["attempts"]
            summary["by_type"][error_type]["successes"] += history["successes"]
            
            # By namespace
            if ns not in summary["by_namespace"]:
                summary["by_namespace"][ns] = {
                    "attempts": 0,
                    "successes": 0
                }
            
            summary["by_namespace"][ns]["attempts"] += history["attempts"]
            summary["by_namespace"][ns]["successes"] += history["successes"]
        
        return summary
    
    def _snapshot_resource_state(self, namespace: str, resource_type: str, 
                               resource_name: str) -> Dict[str, Any]:
        """Take a snapshot of the current state of a Kubernetes resource for potential rollback."""
        try:
            snapshot = {
                'timestamp': time.time(),
                'resource_type': resource_type,
                'resource_name': resource_name,
                'namespace': namespace,
                'state': None
            }
            
            if resource_type == 'pod':
                # Snapshot pod state
                pod = self.k8s_api.read_namespaced_pod(resource_name, namespace)
                snapshot['state'] = {
                    'spec': pod.spec.to_dict(),
                    'metadata': {
                        'labels': pod.metadata.labels,
                        'annotations': pod.metadata.annotations
                    }
                }
            elif resource_type == 'deployment':
                # Snapshot deployment state
                deployment = self.apps_api.read_namespaced_deployment(resource_name, namespace)
                snapshot['state'] = {
                    'spec': deployment.spec.to_dict(),
                    'metadata': {
                        'labels': deployment.metadata.labels,
                        'annotations': deployment.metadata.annotations
                    },
                    'replicas': deployment.spec.replicas
                }
            elif resource_type == 'service':
                # Snapshot service state
                service = self.k8s_api.read_namespaced_service(resource_name, namespace)
                snapshot['state'] = {
                    'spec': service.spec.to_dict(),
                    'metadata': {
                        'labels': service.metadata.labels,
                        'annotations': service.metadata.annotations
                    }
                }
            # Add other resource types as needed
            
            # Store the snapshot
            resource_key = f"{namespace}/{resource_type}/{resource_name}"
            self.state_snapshots[resource_key] = snapshot
            self.logging.log_event(
                "state_snapshot_created",
                f"Created state snapshot for {resource_type}/{resource_name}",
                namespace,
                resource_name
            )
            return snapshot
            
        except Exception as e:
            self.logging.log_event(
                "state_snapshot_failed",
                f"Failed to create state snapshot: {str(e)}",
                namespace,
                resource_name
            )
            return None
    
    def _rollback_resource(self, namespace: str, resource_type: str, 
                         resource_name: str) -> bool:
        """Roll back a resource to its previous state after a failed remediation."""
        resource_key = f"{namespace}/{resource_type}/{resource_name}"
        
        if resource_key not in self.state_snapshots:
            self.logging.log_event(
                "rollback_failed",
                f"No snapshot found for {resource_type}/{resource_name}",
                namespace,
                resource_name
            )
            return False
            
        snapshot = self.state_snapshots[resource_key]
        
        try:
            if resource_type == 'pod':
                # For pods, we can't directly update the spec, so delete and recreate
                try:
                    # Delete the pod
                    self.k8s_api.delete_namespaced_pod(resource_name, namespace)
                    # Wait for the pod to be deleted
                    time.sleep(5)
                    # Create a new pod with the previous spec
                    pod_manifest = {
                        'apiVersion': 'v1',
                        'kind': 'Pod',
                        'metadata': {
                            'name': resource_name,
                            'namespace': namespace,
                            'labels': snapshot['state']['metadata']['labels'],
                            'annotations': snapshot['state']['metadata']['annotations']
                        },
                        'spec': snapshot['state']['spec']
                    }
                    self.k8s_api.create_namespaced_pod(namespace, pod_manifest)
                except Exception as e:
                    self.logging.log_event(
                        "rollback_pod_failed",
                        f"Failed to recreate pod: {str(e)}",
                        namespace,
                        resource_name
                    )
                    return False
                    
            elif resource_type == 'deployment':
                # For deployments, we can patch the spec
                try:
                    patch = {
                        'spec': snapshot['state']['spec'],
                        'metadata': {
                            'labels': snapshot['state']['metadata']['labels'],
                            'annotations': snapshot['state']['metadata']['annotations']
                        }
                    }
                    self.apps_api.patch_namespaced_deployment(
                        resource_name, 
                        namespace, 
                        patch
                    )
                except Exception as e:
                    self.logging.log_event(
                        "rollback_deployment_failed",
                        f"Failed to patch deployment: {str(e)}",
                        namespace,
                        resource_name
                    )
                    return False
                    
            elif resource_type == 'service':
                # For services, we need to delete and recreate
                try:
                    # Delete the service
                    self.k8s_api.delete_namespaced_service(resource_name, namespace)
                    # Wait for the service to be deleted
                    time.sleep(2)
                    # Create a new service with the previous spec
                    service_manifest = {
                        'apiVersion': 'v1',
                        'kind': 'Service',
                        'metadata': {
                            'name': resource_name,
                            'namespace': namespace,
                            'labels': snapshot['state']['metadata']['labels'],
                            'annotations': snapshot['state']['metadata']['annotations']
                        },
                        'spec': snapshot['state']['spec']
                    }
                    self.k8s_api.create_namespaced_service(namespace, service_manifest)
                except Exception as e:
                    self.logging.log_event(
                        "rollback_service_failed",
                        f"Failed to recreate service: {str(e)}",
                        namespace,
                        resource_name
                    )
                    return False
            # Add other resource types as needed
            
            # Log the rollback
            self.logging.log_event(
                "rollback_successful",
                f"Successfully rolled back {resource_type}/{resource_name} to previous state",
                namespace,
                resource_name
            )
            return True
            
        except Exception as e:
            self.logging.log_event(
                "rollback_failed",
                f"Failed to roll back resource: {str(e)}",
                namespace,
                resource_name
            )
            return False
    
    def _get_deployment_from_pod(self, namespace: str, pod_name: str) -> Optional[str]:
        """Get the deployment name for a pod."""
        try:
            # Get the pod
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Check owner references
            if pod.metadata.owner_references:
                for ref in pod.metadata.owner_references:
                    if ref.kind == 'ReplicaSet':
                        # Get the ReplicaSet
                        rs = self.apps_api.read_namespaced_replica_set(ref.name, namespace)
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
            self.logging.log_event(
                "get_deployment_error",
                f"Error getting deployment for pod {pod_name}: {str(e)}",
                namespace,
                pod_name
            )
            return None 