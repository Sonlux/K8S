#!/usr/bin/env python3

import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from kubernetes import client
from .logging_observability import LoggingObservability

class MetricsCollection:
    """Metrics collection system for Kubernetes resources."""
    
    def __init__(self, k8s_api: client.CoreV1Api,
                 metrics_api: client.CustomObjectsApi,
                 logging: LoggingObservability):
        """Initialize metrics collection system."""
        self.k8s_api = k8s_api
        self.metrics_api = metrics_api
        self.logging = logging
        
        # Metrics history
        self.metrics_history = {}
        
        # Collection intervals
        self.intervals = {
            "pod": 60,  # 1 minute
            "node": 300,  # 5 minutes
            "namespace": 600  # 10 minutes
        }
    
    def collect_pod_metrics(self, namespace: str, pod_name: str) -> Dict[str, Any]:
        """Collect metrics for a pod."""
        try:
            # Get pod
            pod = self.k8s_api.read_namespaced_pod(pod_name, namespace)
            
            # Get container metrics
            metrics = {}
            for container in pod.spec.containers:
                container_name = container.name
                
                try:
                    # Get container metrics from metrics-server
                    container_metrics = self.metrics_api.get_namespaced_custom_object(
                        "metrics.k8s.io", "v1beta1", namespace,
                        "pods", f"{pod_name}",
                        container_name
                    )
                    
                    # Extract metrics
                    metrics[container_name] = {
                        "cpu": container_metrics.get("usage", {}).get("cpu", "0"),
                        "memory": container_metrics.get("usage", {}).get("memory", "0"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Update history
                    self._update_metrics_history(
                        "pod", namespace, pod_name, container_name,
                        metrics[container_name]
                    )
                    
                except client.rest.ApiException as e:
                    if e.status == 404:  # Metrics not available
                        self.logging.log_event(
                            "metrics_not_available",
                            f"Metrics not available for container {container_name}",
                            namespace,
                            pod_name
                        )
                    else:
                        raise
            
            return metrics
        
        except Exception as e:
            self.logging.log_event(
                "metrics_collection_failed",
                f"Failed to collect pod metrics: {str(e)}",
                namespace,
                pod_name
            )
            return {}
    
    def collect_node_metrics(self, node_name: str) -> Dict[str, Any]:
        """Collect metrics for a node."""
        try:
            # Get node metrics from metrics-server
            node_metrics = self.metrics_api.get_cluster_custom_object(
                "metrics.k8s.io", "v1beta1", "nodes", node_name
            )
            
            # Extract metrics
            metrics = {
                "cpu": node_metrics.get("usage", {}).get("cpu", "0"),
                "memory": node_metrics.get("usage", {}).get("memory", "0"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update history
            self._update_metrics_history(
                "node", "cluster", node_name, "node",
                metrics
            )
            
            return metrics
        
        except Exception as e:
            self.logging.log_event(
                "metrics_collection_failed",
                f"Failed to collect node metrics: {str(e)}",
                "cluster",
                node_name
            )
            return {}
    
    def collect_namespace_metrics(self, namespace: str) -> Dict[str, Any]:
        """Collect metrics for a namespace."""
        try:
            # Get all pods in namespace
            pods = self.k8s_api.list_namespaced_pod(namespace)
            
            # Collect metrics for each pod
            namespace_metrics = {
                "cpu": "0",
                "memory": "0",
                "pod_count": len(pods.items),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            for pod in pods.items:
                pod_metrics = self.collect_pod_metrics(namespace, pod.metadata.name)
                
                # Aggregate pod metrics
                for container_metrics in pod_metrics.values():
                    namespace_metrics["cpu"] = self._add_resource_quantities(
                        namespace_metrics["cpu"],
                        container_metrics["cpu"]
                    )
                    namespace_metrics["memory"] = self._add_resource_quantities(
                        namespace_metrics["memory"],
                        container_metrics["memory"]
                    )
            
            # Update history
            self._update_metrics_history(
                "namespace", namespace, namespace, "namespace",
                namespace_metrics
            )
            
            return namespace_metrics
        
        except Exception as e:
            self.logging.log_event(
                "metrics_collection_failed",
                f"Failed to collect namespace metrics: {str(e)}",
                namespace,
                namespace
            )
            return {}
    
    def _update_metrics_history(self, resource_type: str, namespace: str,
                              resource_name: str, container_name: str,
                              metrics: Dict[str, Any]):
        """Update metrics history."""
        key = f"{resource_type}/{namespace}/{resource_name}/{container_name}"
        
        if key not in self.metrics_history:
            self.metrics_history[key] = []
        
        # Add new metrics
        self.metrics_history[key].append(metrics)
        
        # Keep last 24 hours of metrics
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.metrics_history[key] = [
            m for m in self.metrics_history[key]
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
    
    def _add_resource_quantities(self, q1: str, q2: str) -> str:
        """Add two resource quantities."""
        try:
            # Parse quantities
            v1, u1 = self._parse_quantity(q1)
            v2, u2 = self._parse_quantity(q2)
            
            # Convert to same unit
            if u1 != u2:
                if u1 == "Mi" and u2 == "Gi":
                    v1 = v1 / 1024
                    u1 = "Gi"
                elif u1 == "Gi" and u2 == "Mi":
                    v2 = v2 / 1024
                    u2 = "Gi"
                elif u1 == "m" and u2 == "":
                    v1 = v1 / 1000
                    u1 = ""
                elif u1 == "" and u2 == "m":
                    v2 = v2 / 1000
                    u2 = ""
            
            # Add values
            total = v1 + v2
            
            # Format result
            if u1 == "Gi":
                if total >= 1024:
                    return f"{total / 1024}Ti"
                return f"{total}Gi"
            elif u1 == "Mi":
                if total >= 1024:
                    return f"{total / 1024}Gi"
                return f"{total}Mi"
            elif u1 == "":
                if total >= 1000:
                    return f"{total / 1000}"
                return f"{total}m"
            
            return q1
        
        except (ValueError, IndexError):
            return q1
    
    def _parse_quantity(self, quantity: str) -> Tuple[float, str]:
        """Parse a resource quantity."""
        try:
            # Handle CPU quantities
            if quantity.endswith("m"):
                return float(quantity[:-1]), "m"
            elif quantity.isdigit():
                return float(quantity), ""
            
            # Handle memory quantities
            value = float(quantity[:-2])
            unit = quantity[-2:]
            
            return value, unit
        
        except (ValueError, IndexError):
            return 0.0, ""
    
    def get_metrics_history(self, resource_type: str, namespace: str,
                          resource_name: str,
                          container_name: Optional[str] = None,
                          hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history."""
        key = f"{resource_type}/{namespace}/{resource_name}"
        if container_name:
            key = f"{key}/{container_name}"
        
        if key not in self.metrics_history:
            return []
        
        # Filter by time range
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history[key]
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
    
    def get_metrics_summary(self, resource_type: str, namespace: str,
                          resource_name: str,
                          container_name: Optional[str] = None,
                          hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary."""
        metrics = self.get_metrics_history(
            resource_type, namespace, resource_name,
            container_name, hours
        )
        
        if not metrics:
            return {}
        
        # Calculate summary statistics
        summary = {
            "count": len(metrics),
            "cpu": {
                "min": "0",
                "max": "0",
                "avg": "0"
            },
            "memory": {
                "min": "0",
                "max": "0",
                "avg": "0"
            }
        }
        
        # Calculate min/max/avg
        cpu_values = []
        memory_values = []
        
        for metric in metrics:
            cpu_values.append(self._parse_quantity(metric["cpu"])[0])
            memory_values.append(self._parse_quantity(metric["memory"])[0])
        
        if cpu_values:
            summary["cpu"]["min"] = f"{min(cpu_values)}m"
            summary["cpu"]["max"] = f"{max(cpu_values)}m"
            summary["cpu"]["avg"] = f"{sum(cpu_values) / len(cpu_values)}m"
        
        if memory_values:
            summary["memory"]["min"] = f"{min(memory_values)}Mi"
            summary["memory"]["max"] = f"{max(memory_values)}Mi"
            summary["memory"]["avg"] = f"{sum(memory_values) / len(memory_values)}Mi"
        
        return summary 