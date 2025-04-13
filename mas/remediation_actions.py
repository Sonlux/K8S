#!/usr/bin/env python3

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from kubernetes import client
from .logging_observability import LoggingObservability

class RemediationActions:
    """Remediation actions for the Multi-Agent System."""
    
    def __init__(self, k8s_api: client.CoreV1Api, apps_api: client.AppsV1Api,
                 logging: LoggingObservability):
        """Initialize remediation actions."""
        self.logger = logging.getLogger("mas-remediation")
        self.k8s_api = k8s_api
        self.apps_api = apps_api
        self.logging = logging
        
        # Track remediation attempts
        self.remediation_history = {}
        self.remediation_cooldown = timedelta(minutes=5)
    
    def adjust_resource_limits(self, namespace: str, pod_name: str,
                             container: str, params: Dict[str, Any]) -> bool:
        """Adjust resource limits for a container."""
        try:
            # Get the pod's owner reference (deployment, statefulset, etc.)
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            owner_ref = self._get_owner_reference(pod)
            
            if not owner_ref:
                self.logger.error(f"No owner reference found for pod {pod_name}")
                return False
            
            # Get the deployment/statefulset
            if owner_ref["kind"] == "Deployment":
                resource = self.apps_api.read_namespaced_deployment(
                    owner_ref["name"], namespace)
                template = resource.spec.template
            elif owner_ref["kind"] == "StatefulSet":
                resource = self.apps_api.read_namespaced_stateful_set(
                    owner_ref["name"], namespace)
                template = resource.spec.template
            else:
                self.logger.error(f"Unsupported owner kind: {owner_ref['kind']}")
                return False
            
            # Update container resources
            for container_spec in template.spec.containers:
                if container_spec.name == container:
                    container_spec.resources = client.V1ResourceRequirements(
                        limits={"memory": params["memory_limit"]},
                        requests={"memory": params["memory_request"]}
                    )
                    break
            
            # Update the deployment/statefulset
            if owner_ref["kind"] == "Deployment":
                self.apps_api.patch_namespaced_deployment(
                    owner_ref["name"], namespace, resource)
            else:
                self.apps_api.patch_namespaced_stateful_set(
                    owner_ref["name"], namespace, resource)
            
            self.logging.log_event(
                event_type="resource_adjusted",
                message=f"Adjusted resource limits for {container} in {pod_name}",
                namespace=namespace,
                pod_name=pod_name,
                metadata={
                    "container": container,
                    "memory_limit": params["memory_limit"],
                    "memory_request": params["memory_request"]
                }
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to adjust resource limits: {str(e)}")
            return False
    
    def restart_pod(self, namespace: str, pod_name: str,
                   params: Dict[str, Any]) -> bool:
        """Restart a pod with grace period."""
        try:
            self.k8s_api.delete_namespaced_pod(
                pod_name, namespace,
                grace_period_seconds=params["grace_period"]
            )
            
            self.logging.log_event(
                event_type="pod_restarted",
                message=f"Restarted pod {pod_name}",
                namespace=namespace,
                pod_name=pod_name,
                metadata={"grace_period": params["grace_period"]}
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restart pod: {str(e)}")
            return False
    
    def fix_image_pull(self, namespace: str, pod_name: str,
                      params: Dict[str, Any]) -> bool:
        """Fix image pull issues."""
        try:
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            owner_ref = self._get_owner_reference(pod)
            
            if not owner_ref:
                self.logger.error(f"No owner reference found for pod {pod_name}")
                return False
            
            # Get the deployment/statefulset
            if owner_ref["kind"] == "Deployment":
                resource = self.apps_api.read_namespaced_deployment(
                    owner_ref["name"], namespace)
                template = resource.spec.template
            elif owner_ref["kind"] == "StatefulSet":
                resource = self.apps_api.read_namespaced_stateful_set(
                    owner_ref["name"], namespace)
                template = resource.spec.template
            else:
                self.logger.error(f"Unsupported owner kind: {owner_ref['kind']}")
                return False
            
            # Add imagePullPolicy: Always
            for container in template.spec.containers:
                container.image_pull_policy = "Always"
            
            # Update the deployment/statefulset
            if owner_ref["kind"] == "Deployment":
                self.apps_api.patch_namespaced_deployment(
                    owner_ref["name"], namespace, resource)
            else:
                self.apps_api.patch_namespaced_stateful_set(
                    owner_ref["name"], namespace, resource)
            
            self.logging.log_event(
                event_type="image_pull_fixed",
                message=f"Fixed image pull policy for {pod_name}",
                namespace=namespace,
                pod_name=pod_name
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to fix image pull: {str(e)}")
            return False
    
    def check_readiness(self, namespace: str, pod_name: str,
                       params: Dict[str, Any]) -> bool:
        """Check pod readiness with timeout."""
        try:
            start_time = datetime.utcnow()
            timeout = timedelta(seconds=params["timeout"])
            interval = timedelta(seconds=params["check_interval"])
            
            while datetime.utcnow() - start_time < timeout:
                pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
                
                # Check if pod is ready
                if pod.status.conditions:
                    ready_condition = next(
                        (c for c in pod.status.conditions if c.type == "Ready"),
                        None
                    )
                    if ready_condition and ready_condition.status == "True":
                        self.logging.log_event(
                            event_type="pod_ready",
                            message=f"Pod {pod_name} is now ready",
                            namespace=namespace,
                            pod_name=pod_name
                        )
                        return True
                
                # Wait for next check
                time.sleep(interval.total_seconds())
            
            self.logger.error(f"Pod {pod_name} did not become ready within timeout")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check readiness: {str(e)}")
            return False
    
    def investigate_logs(self, namespace: str, pod_name: str,
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Investigate pod logs for errors."""
        try:
            logs = self.logging.get_pod_logs(
                namespace, pod_name,
                tail_lines=params["log_lines"]
            )
            
            # Analyze logs for error pattern
            error_lines = []
            for line in logs.split('\n'):
                if params["error_pattern"] in line:
                    error_lines.append(line.strip())
            
            investigation_result = {
                "pod": pod_name,
                "namespace": namespace,
                "error_lines": error_lines,
                "total_lines_analyzed": len(logs.split('\n')),
                "error_count": len(error_lines)
            }
            
            self.logging.log_event(
                event_type="logs_investigated",
                message=f"Investigated logs for {pod_name}",
                namespace=namespace,
                pod_name=pod_name,
                metadata=investigation_result
            )
            
            return investigation_result
            
        except Exception as e:
            self.logger.error(f"Failed to investigate logs: {str(e)}")
            return {
                "pod": pod_name,
                "namespace": namespace,
                "error": str(e)
            }
    
    def _get_owner_reference(self, pod: client.V1Pod) -> Optional[Dict[str, str]]:
        """Get the owner reference of a pod."""
        if not pod.metadata.owner_references:
            return None
        
        owner = pod.metadata.owner_references[0]
        return {
            "kind": owner.kind,
            "name": owner.name,
            "uid": owner.uid
        }
    
    def can_remediate(self, namespace: str, pod_name: str,
                     action: str) -> bool:
        """Check if remediation can be performed (cooldown period)."""
        key = f"{namespace}/{pod_name}/{action}"
        now = datetime.utcnow()
        
        if key in self.remediation_history:
            last_attempt = self.remediation_history[key]
            if now - last_attempt < self.remediation_cooldown:
                return False
        
        self.remediation_history[key] = now
        return True 