#!/usr/bin/env python3

import logging
import time
from typing import Dict, List, Any, Optional
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger("mas-keda")

class KedaIntegration:
    """Integration with KEDA for event-driven autoscaling."""
    
    def __init__(self, k8s_api: client.CustomObjectsApi = None):
        """Initialize KEDA integration."""
        self.logger = logging.getLogger("mas-keda")
        
        # Setup Kubernetes client if not provided
        if k8s_api is None:
            try:
                config.load_kube_config()
                self.k8s_api = client.CustomObjectsApi()
            except Exception as e:
                self.logger.error(f"Failed to initialize Kubernetes client: {str(e)}")
                self.k8s_api = None
        else:
            self.k8s_api = k8s_api
        
        # KEDA API group and version
        self.keda_group = "keda.sh"
        self.keda_version = "v1alpha1"
        
        # Track ScaledObjects
        self.scaled_objects = {}
        self.last_refresh = 0
        self.refresh_interval = 60  # seconds
    
    def refresh_scaled_objects(self):
        """Refresh the list of ScaledObjects from the cluster."""
        if time.time() - self.last_refresh < self.refresh_interval:
            return
        
        try:
            # Get all ScaledObjects across all namespaces
            scaled_objects = self.k8s_api.list_cluster_custom_object(
                group=self.keda_group,
                version=self.keda_version,
                plural="scaledobjects"
            )
            
            # Update our cache
            self.scaled_objects = {
                f"{obj['metadata']['namespace']}/{obj['metadata']['name']}": obj
                for obj in scaled_objects.get('items', [])
            }
            
            self.last_refresh = time.time()
            self.logger.info(f"Refreshed {len(self.scaled_objects)} ScaledObjects")
            
        except ApiException as e:
            self.logger.error(f"Failed to refresh ScaledObjects: {str(e)}")
    
    def get_scaled_object(self, namespace: str, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific ScaledObject by namespace and name."""
        self.refresh_scaled_objects()
        return self.scaled_objects.get(f"{namespace}/{name}")
    
    def create_scaled_object(self, namespace: str, name: str, deployment_name: str, 
                            min_replicas: int = 1, max_replicas: int = 10,
                            polling_interval: int = 30, cooldown_period: int = 300,
                            triggers: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new ScaledObject for a deployment."""
        if triggers is None:
            triggers = [
                {
                    "type": "cpu",
                    "metadata": {
                        "type": "Utilization",
                        "value": "80"
                    }
                }
            ]
        
        scaled_object = {
            "apiVersion": f"{self.keda_group}/{self.keda_version}",
            "kind": "ScaledObject",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "name": deployment_name
                },
                "minReplicaCount": min_replicas,
                "maxReplicaCount": max_replicas,
                "pollingInterval": polling_interval,
                "cooldownPeriod": cooldown_period,
                "triggers": triggers
            }
        }
        
        try:
            result = self.k8s_api.create_namespaced_custom_object(
                group=self.keda_group,
                version=self.keda_version,
                namespace=namespace,
                plural="scaledobjects",
                body=scaled_object
            )
            
            self.logger.info(f"Created ScaledObject {namespace}/{name} for deployment {deployment_name}")
            return result
            
        except ApiException as e:
            self.logger.error(f"Failed to create ScaledObject: {str(e)}")
            return {"error": str(e)}
    
    def update_scaled_object(self, namespace: str, name: str, 
                            min_replicas: Optional[int] = None,
                            max_replicas: Optional[int] = None,
                            polling_interval: Optional[int] = None,
                            cooldown_period: Optional[int] = None,
                            triggers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Update an existing ScaledObject."""
        # Get the current ScaledObject
        scaled_object = self.get_scaled_object(namespace, name)
        if not scaled_object:
            self.logger.error(f"ScaledObject {namespace}/{name} not found")
            return {"error": "ScaledObject not found"}
        
        # Update fields if provided
        if min_replicas is not None:
            scaled_object["spec"]["minReplicaCount"] = min_replicas
        
        if max_replicas is not None:
            scaled_object["spec"]["maxReplicaCount"] = max_replicas
        
        if polling_interval is not None:
            scaled_object["spec"]["pollingInterval"] = polling_interval
        
        if cooldown_period is not None:
            scaled_object["spec"]["cooldownPeriod"] = cooldown_period
        
        if triggers is not None:
            scaled_object["spec"]["triggers"] = triggers
        
        try:
            result = self.k8s_api.patch_namespaced_custom_object(
                group=self.keda_group,
                version=self.keda_version,
                namespace=namespace,
                plural="scaledobjects",
                name=name,
                body=scaled_object
            )
            
            self.logger.info(f"Updated ScaledObject {namespace}/{name}")
            return result
            
        except ApiException as e:
            self.logger.error(f"Failed to update ScaledObject: {str(e)}")
            return {"error": str(e)}
    
    def delete_scaled_object(self, namespace: str, name: str) -> Dict[str, Any]:
        """Delete a ScaledObject."""
        try:
            result = self.k8s_api.delete_namespaced_custom_object(
                group=self.keda_group,
                version=self.keda_version,
                namespace=namespace,
                plural="scaledobjects",
                name=name
            )
            
            self.logger.info(f"Deleted ScaledObject {namespace}/{name}")
            return result
            
        except ApiException as e:
            self.logger.error(f"Failed to delete ScaledObject: {str(e)}")
            return {"error": str(e)}
    
    def get_scaled_object_status(self, namespace: str, name: str) -> Dict[str, Any]:
        """Get the status of a ScaledObject."""
        try:
            result = self.k8s_api.get_namespaced_custom_object_status(
                group=self.keda_group,
                version=self.keda_version,
                namespace=namespace,
                plural="scaledobjects",
                name=name
            )
            
            return result
            
        except ApiException as e:
            self.logger.error(f"Failed to get ScaledObject status: {str(e)}")
            return {"error": str(e)}
    
    def create_cpu_scaled_object(self, namespace: str, name: str, deployment_name: str,
                                min_replicas: int = 1, max_replicas: int = 10,
                                target_cpu_percentage: int = 80) -> Dict[str, Any]:
        """Create a ScaledObject with CPU-based scaling."""
        triggers = [
            {
                "type": "cpu",
                "metadata": {
                    "type": "Utilization",
                    "value": str(target_cpu_percentage)
                }
            }
        ]
        
        return self.create_scaled_object(
            namespace=namespace,
            name=name,
            deployment_name=deployment_name,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            triggers=triggers
        )
    
    def create_memory_scaled_object(self, namespace: str, name: str, deployment_name: str,
                                   min_replicas: int = 1, max_replicas: int = 10,
                                   target_memory_percentage: int = 80) -> Dict[str, Any]:
        """Create a ScaledObject with memory-based scaling."""
        triggers = [
            {
                "type": "memory",
                "metadata": {
                    "type": "Utilization",
                    "value": str(target_memory_percentage)
                }
            }
        ]
        
        return self.create_scaled_object(
            namespace=namespace,
            name=name,
            deployment_name=deployment_name,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            triggers=triggers
        )
    
    def create_prometheus_scaled_object(self, namespace: str, name: str, deployment_name: str,
                                       min_replicas: int = 1, max_replicas: int = 10,
                                       query: str = "", server_address: str = "http://prometheus-server.monitoring:9090",
                                       threshold: str = "100") -> Dict[str, Any]:
        """Create a ScaledObject with Prometheus-based scaling."""
        triggers = [
            {
                "type": "prometheus",
                "metadata": {
                    "serverAddress": server_address,
                    "query": query,
                    "threshold": threshold
                }
            }
        ]
        
        return self.create_scaled_object(
            namespace=namespace,
            name=name,
            deployment_name=deployment_name,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            triggers=triggers
        )
    
    def create_kafka_scaled_object(self, namespace: str, name: str, deployment_name: str,
                                  min_replicas: int = 1, max_replicas: int = 10,
                                  bootstrap_servers: str = "kafka:9092",
                                  consumer_group: str = "my-group",
                                  topic: str = "my-topic",
                                  lag_threshold: str = "10") -> Dict[str, Any]:
        """Create a ScaledObject with Kafka-based scaling."""
        triggers = [
            {
                "type": "kafka",
                "metadata": {
                    "bootstrapServers": bootstrap_servers,
                    "consumerGroup": consumer_group,
                    "topic": topic,
                    "lagThreshold": lag_threshold
                }
            }
        ]
        
        return self.create_scaled_object(
            namespace=namespace,
            name=name,
            deployment_name=deployment_name,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            triggers=triggers
        ) 