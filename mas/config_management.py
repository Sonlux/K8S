#!/usr/bin/env python3

import os
import yaml
import json
from typing import Dict, Any, Optional
from kubernetes import client

class ConfigManagement:
    """Configuration management system for Kubernetes resources."""
    
    def __init__(self, k8s_api: client.CoreV1Api):
        """Initialize configuration management system."""
        self.k8s_api = k8s_api
        
        # Default configuration
        self.default_config = {
            "monitoring": {
                "interval": 60,  # seconds
                "metrics_retention": 24,  # hours
                "event_retention": 24,  # hours
                "max_history_size": 1000
            },
            "remediation": {
                "cooldown_period": 300,  # seconds
                "max_retries": 3,
                "rollback_enabled": True,
                "auto_scale": True
            },
            "resources": {
                "cpu_request": "100m",
                "cpu_limit": "500m",
                "memory_request": "128Mi",
                "memory_limit": "512Mi"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "/var/log/mas/mas.log"
            }
        }
        
        # Current configuration
        self.config = self.default_config.copy()
        
        # Load configuration from ConfigMap
        self._load_config()
    
    def _load_config(self):
        """Load configuration from ConfigMap."""
        try:
            # Get ConfigMap
            config_map = self.k8s_api.read_namespaced_config_map(
                "mas-config",
                "default"
            )
            
            # Parse configuration
            if "config.yaml" in config_map.data:
                config = yaml.safe_load(config_map.data["config.yaml"])
                self.config.update(config)
            
            elif "config.json" in config_map.data:
                config = json.loads(config_map.data["config.json"])
                self.config.update(config)
        
        except client.rest.ApiException as e:
            if e.status == 404:
                # ConfigMap not found, create it
                self._create_config_map()
            else:
                raise
    
    def _create_config_map(self):
        """Create ConfigMap with default configuration."""
        try:
            # Create ConfigMap
            config_map = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name="mas-config",
                    namespace="default"
                ),
                data={
                    "config.yaml": yaml.dump(self.config)
                }
            )
            
            # Create ConfigMap
            self.k8s_api.create_namespaced_config_map(
                "default",
                config_map
            )
        
        except Exception as e:
            raise Exception(f"Failed to create ConfigMap: {str(e)}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration."""
        try:
            # Update configuration
            self.config.update(new_config)
            
            # Update ConfigMap
            config_map = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name="mas-config",
                    namespace="default"
                ),
                data={
                    "config.yaml": yaml.dump(self.config)
                }
            )
            
            # Update ConfigMap
            self.k8s_api.patch_namespaced_config_map(
                "mas-config",
                "default",
                config_map
            )
        
        except Exception as e:
            raise Exception(f"Failed to update configuration: {str(e)}")
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration."""
        if section:
            return self.config.get(section, {})
        return self.config
    
    def reset_config(self):
        """Reset configuration to defaults."""
        try:
            # Reset configuration
            self.config = self.default_config.copy()
            
            # Update ConfigMap
            config_map = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name="mas-config",
                    namespace="default"
                ),
                data={
                    "config.yaml": yaml.dump(self.config)
                }
            )
            
            # Update ConfigMap
            self.k8s_api.patch_namespaced_config_map(
                "mas-config",
                "default",
                config_map
            )
        
        except Exception as e:
            raise Exception(f"Failed to reset configuration: {str(e)}")
    
    def export_config(self, file_path: str):
        """Export configuration to a file."""
        try:
            with open(file_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        
        except Exception as e:
            raise Exception(f"Failed to export configuration: {str(e)}")
    
    def import_config(self, file_path: str):
        """Import configuration from a file."""
        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Update configuration
            self.update_config(config)
        
        except Exception as e:
            raise Exception(f"Failed to import configuration: {str(e)}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        try:
            # Check required sections
            required_sections = ["monitoring", "remediation", "resources", "logging"]
            for section in required_sections:
                if section not in config:
                    return False
            
            # Check monitoring section
            monitoring = config["monitoring"]
            if not all(k in monitoring for k in ["interval", "metrics_retention", "event_retention"]):
                return False
            
            # Check remediation section
            remediation = config["remediation"]
            if not all(k in remediation for k in ["cooldown_period", "max_retries", "rollback_enabled"]):
                return False
            
            # Check resources section
            resources = config["resources"]
            if not all(k in resources for k in ["cpu_request", "cpu_limit", "memory_request", "memory_limit"]):
                return False
            
            # Check logging section
            logging = config["logging"]
            if not all(k in logging for k in ["level", "format", "file"]):
                return False
            
            return True
        
        except Exception:
            return False 