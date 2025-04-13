#!/usr/bin/env python3

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from kubernetes import client
from .logging_observability import LoggingObservability

class ErrorDetector:
    """Error detection system for Kubernetes resources."""
    
    def __init__(self, k8s_api: client.CoreV1Api,
                 logging: LoggingObservability):
        """Initialize error detector."""
        self.k8s_api = k8s_api
        self.logging = logging
        
        # Error patterns
        self.error_patterns = {
            "oom": [
                r"OutOfMemory",
                r"memory limit exceeded",
                r"container killed due to OOM"
            ],
            "crash_loop": [
                r"CrashLoopBackOff",
                r"Error: exit status \d+",
                r"container terminated"
            ],
            "image_pull": [
                r"ImagePullBackOff",
                r"Failed to pull image",
                r"image not found"
            ],
            "network": [
                r"connection refused",
                r"connection timed out",
                r"no route to host"
            ],
            "resource": [
                r"resource quota exceeded",
                r"insufficient resources",
                r"resource limit exceeded"
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {
            error_type: [re.compile(pattern) for pattern in patterns]
            for error_type, patterns in self.error_patterns.items()
        }
    
    def detect_errors(self, pod_name: str, namespace: str) -> List[Dict[str, Any]]:
        """Detect errors in a pod."""
        errors = []
        
        try:
            # Get pod status
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Check container statuses
            if pod.status.container_statuses:
                for container in pod.status.container_statuses:
                    container_errors = self._check_container_status(
                        container, pod_name, namespace
                    )
                    errors.extend(container_errors)
            
            # Check pod events
            pod_events = self._get_pod_events(pod_name, namespace)
            event_errors = self._analyze_pod_events(pod_events, pod_name, namespace)
            errors.extend(event_errors)
            
            # Check pod logs
            logs = self.logging.get_pod_logs(namespace, pod_name)
            log_errors = self._analyze_pod_logs(logs, pod_name, namespace)
            errors.extend(log_errors)
            
        except Exception as e:
            self.logging.log_event(
                "error_detection_failed",
                f"Failed to detect errors: {str(e)}",
                namespace,
                pod_name
            )
        
        return errors
    
    def _check_container_status(self, container: Any,
                              pod_name: str, namespace: str) -> List[Dict[str, Any]]:
        """Check container status for errors."""
        errors = []
        
        # Check state
        if container.state.waiting:
            error = self._analyze_waiting_state(
                container.state.waiting,
                container.name,
                pod_name,
                namespace
            )
            if error:
                errors.append(error)
        
        elif container.state.terminated:
            error = self._analyze_terminated_state(
                container.state.terminated,
                container.name,
                pod_name,
                namespace
            )
            if error:
                errors.append(error)
        
        # Check restart count
        if container.restart_count > 0:
            errors.append({
                "type": "crash_loop",
                "severity": "warning",
                "message": f"Container {container.name} has restarted {container.restart_count} times",
                "container": container.name,
                "pod": pod_name,
                "namespace": namespace,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return errors
    
    def _analyze_waiting_state(self, state: Any, container_name: str,
                             pod_name: str, namespace: str) -> Optional[Dict[str, Any]]:
        """Analyze container waiting state."""
        if state.reason == "ImagePullBackOff":
            return {
                "type": "image_pull",
                "severity": "error",
                "message": f"Failed to pull image: {state.message}",
                "container": container_name,
                "pod": pod_name,
                "namespace": namespace,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return None
    
    def _analyze_terminated_state(self, state: Any, container_name: str,
                                pod_name: str, namespace: str) -> Optional[Dict[str, Any]]:
        """Analyze container terminated state."""
        if state.exit_code != 0:
            return {
                "type": "crash_loop",
                "severity": "error",
                "message": f"Container terminated with exit code {state.exit_code}: {state.message}",
                "container": container_name,
                "pod": pod_name,
                "namespace": namespace,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return None
    
    def _get_pod_events(self, pod_name: str, namespace: str) -> List[Dict[str, Any]]:
        """Get events related to a pod."""
        try:
            events = self.k8s_api.list_namespaced_event(
                namespace=namespace,
                field_selector=f"involvedObject.name={pod_name}"
            )
            
            return [
                {
                    "type": event.type,
                    "reason": event.reason,
                    "message": event.message,
                    "timestamp": event.last_timestamp.isoformat() if event.last_timestamp else None
                }
                for event in events.items
            ]
        except Exception as e:
            self.logging.log_event(
                "event_retrieval_failed",
                f"Failed to get pod events: {str(e)}",
                namespace,
                pod_name
            )
            return []
    
    def _analyze_pod_events(self, events: List[Dict[str, Any]],
                          pod_name: str, namespace: str) -> List[Dict[str, Any]]:
        """Analyze pod events for errors."""
        errors = []
        
        for event in events:
            if event["type"] == "Warning":
                error_type = self._determine_error_type(event["message"])
                if error_type:
                    errors.append({
                        "type": error_type,
                        "severity": "warning",
                        "message": event["message"],
                        "pod": pod_name,
                        "namespace": namespace,
                        "timestamp": event["timestamp"] or datetime.utcnow().isoformat()
                    })
        
        return errors
    
    def _analyze_pod_logs(self, logs: str, pod_name: str,
                         namespace: str) -> List[Dict[str, Any]]:
        """Analyze pod logs for errors."""
        errors = []
        
        for error_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(logs)
                for match in matches:
                    errors.append({
                        "type": error_type,
                        "severity": "error",
                        "message": match.group(0),
                        "pod": pod_name,
                        "namespace": namespace,
                        "timestamp": datetime.utcnow().isoformat()
                    })
        
        return errors
    
    def _determine_error_type(self, message: str) -> Optional[str]:
        """Determine error type from message."""
        for error_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(message):
                    return error_type
        return None
    
    def get_error_summary(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of detected errors."""
        summary = {
            "total_errors": 0,
            "by_type": {},
            "by_severity": {},
            "by_namespace": {}
        }
        
        # Get all pods
        try:
            pods = self.k8s_api.list_namespaced_pod(
                namespace=namespace if namespace else ""
            )
            
            # Analyze each pod
            for pod in pods.items:
                errors = self.detect_errors(pod.metadata.name, pod.metadata.namespace)
                
                # Update summary
                summary["total_errors"] += len(errors)
                
                for error in errors:
                    # By type
                    error_type = error["type"]
                    if error_type not in summary["by_type"]:
                        summary["by_type"][error_type] = 0
                    summary["by_type"][error_type] += 1
                    
                    # By severity
                    severity = error["severity"]
                    if severity not in summary["by_severity"]:
                        summary["by_severity"][severity] = 0
                    summary["by_severity"][severity] += 1
                    
                    # By namespace
                    ns = error["namespace"]
                    if ns not in summary["by_namespace"]:
                        summary["by_namespace"][ns] = 0
                    summary["by_namespace"][ns] += 1
        
        except Exception as e:
            self.logging.log_event(
                "error_summary_failed",
                f"Failed to generate error summary: {str(e)}",
                namespace or "default",
                "error-detector"
            )
        
        return summary 