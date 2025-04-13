#!/usr/bin/env python3

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from kubernetes import client
from prometheus_client import Counter, Gauge, Histogram

class LoggingObservability:
    """Logging and observability system for Kubernetes resources."""
    
    def __init__(self, k8s_api: client.CoreV1Api):
        """Initialize logging and observability system."""
        self.k8s_api = k8s_api
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
        
        # Event history
        self.event_history = []
        
        # Event types
        self.event_types = {
            "metrics_not_available": "warning",
            "metrics_collection_failed": "error",
            "remediation_started": "info",
            "remediation_succeeded": "info",
            "remediation_failed": "error",
            "pod_restarted": "info",
            "pod_deleted": "info",
            "pod_created": "info",
            "deployment_updated": "info",
            "resource_updated": "info",
            "error_detected": "warning",
            "error_resolved": "info"
        }
        
        # Prometheus metrics
        self.event_counter = Counter(
            'mas_events_total',
            'Total number of events',
            ['event_type', 'namespace']
        )
        self.error_gauge = Gauge(
            'mas_errors',
            'Number of active errors',
            ['severity', 'namespace']
        )
        self.remediation_duration = Histogram(
            'mas_remediation_duration_seconds',
            'Time taken for remediation actions',
            ['action_type']
        )
    
    def log_event(self, event_type: str, message: str,
                 namespace: str, resource_name: str,
                 metadata: Optional[Dict[str, Any]] = None):
        """Log an event."""
        # Create event
        event = {
            "type": event_type,
            "message": message,
            "namespace": namespace,
            "resource_name": resource_name,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to history
        self.event_history.append(event)
        
        # Keep last 1000 events
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
        
        # Log based on event type
        level = self.event_types.get(event_type, "info")
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        # Create Kubernetes event
        try:
            self._create_k8s_event(event)
        except Exception as e:
            self.logger.error(f"Failed to create Kubernetes event: {str(e)}")
        
        # Update Prometheus metrics
        self.event_counter.labels(
            event_type=event_type,
            namespace=namespace
        ).inc()
    
    def _create_k8s_event(self, event: Dict[str, Any]):
        """Create a Kubernetes event."""
        # Create event object
        k8s_event = client.CoreV1Event(
            metadata=client.V1ObjectMeta(
                generate_name=f"{event['type']}-",
                namespace=event["namespace"]
            ),
            type=event["type"].upper(),
            reason=event["type"],
            message=event["message"],
            source=client.V1EventSource(
                component="mas",
                host="mas"
            ),
            first_timestamp=datetime.utcnow(),
            last_timestamp=datetime.utcnow(),
            count=1,
            involved_object=client.V1ObjectReference(
                kind="Pod",
                name=event["resource_name"],
                namespace=event["namespace"]
            )
        )
        
        # Create event
        self.k8s_api.create_namespaced_event(
            event["namespace"],
            k8s_event
        )
    
    def get_event_history(self, event_type: Optional[str] = None,
                         namespace: Optional[str] = None,
                         resource_name: Optional[str] = None,
                         hours: int = 24) -> List[Dict[str, Any]]:
        """Get event history."""
        # Filter events
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        
        if namespace:
            events = [e for e in events if e["namespace"] == namespace]
        
        if resource_name:
            events = [e for e in events if e["resource_name"] == resource_name]
        
        # Filter by time range
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        events = [
            e for e in events
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
        
        return events
    
    def get_event_summary(self, event_type: Optional[str] = None,
                         namespace: Optional[str] = None,
                         resource_name: Optional[str] = None,
                         hours: int = 24) -> Dict[str, Any]:
        """Get event summary."""
        events = self.get_event_history(
            event_type, namespace, resource_name, hours
        )
        
        # Calculate summary
        summary = {
            "total": len(events),
            "by_type": {},
            "by_namespace": {},
            "by_resource": {}
        }
        
        # Count events
        for event in events:
            # Count by type
            event_type = event["type"]
            if event_type not in summary["by_type"]:
                summary["by_type"][event_type] = 0
            summary["by_type"][event_type] += 1
            
            # Count by namespace
            namespace = event["namespace"]
            if namespace not in summary["by_namespace"]:
                summary["by_namespace"][namespace] = 0
            summary["by_namespace"][namespace] += 1
            
            # Count by resource
            resource = event["resource_name"]
            if resource not in summary["by_resource"]:
                summary["by_resource"][resource] = 0
            summary["by_resource"][resource] += 1
        
        return summary
    
    def export_events(self, file_path: str):
        """Export events to a file."""
        try:
            with open(file_path, "w") as f:
                json.dump(self.event_history, f, indent=2)
            
            self.logger.info(f"Events exported to {file_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to export events: {str(e)}")
    
    def import_events(self, file_path: str):
        """Import events from a file."""
        try:
            with open(file_path, "r") as f:
                events = json.load(f)
            
            # Add events to history
            self.event_history.extend(events)
            
            # Keep last 1000 events
            if len(self.event_history) > 1000:
                self.event_history = self.event_history[-1000:]
            
            self.logger.info(f"Events imported from {file_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to import events: {str(e)}")
    
    def clear_event_history(self):
        """Clear event history."""
        self.event_history = []
        self.logger.info("Event history cleared")
    
    def update_error_gauge(self, severity: str, namespace: str,
                          count: int) -> None:
        """Update error gauge for a namespace."""
        self.error_gauge.labels(
            severity=severity,
            namespace=namespace
        ).set(count)
    
    def record_remediation_duration(self, action_type: str,
                                  duration: float) -> None:
        """Record remediation action duration."""
        self.remediation_duration.labels(
            action_type=action_type
        ).observe(duration)
    
    def get_pod_logs(self, namespace: str, pod_name: str,
                    tail_lines: int = 100) -> str:
        """Get pod logs."""
        try:
            return self.k8s_api.read_namespaced_pod_log(
                pod_name, namespace,
                tail_lines=tail_lines
            )
        except Exception as e:
            self.logger.error(f"Failed to get logs for {pod_name}: {str(e)}")
            return ""
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        from prometheus_client import generate_latest
        return generate_latest().decode('utf-8')
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of current errors."""
        summary = {
            "total_errors": sum(
                self.error_gauge._value.get()
            ),
            "by_severity": {},
            "by_namespace": {}
        }
        
        # Aggregate by severity
        for labels, value in self.error_gauge._value.items():
            severity = labels[0]
            namespace = labels[1]
            
            if severity not in summary["by_severity"]:
                summary["by_severity"][severity] = 0
            summary["by_severity"][severity] += value
            
            if namespace not in summary["by_namespace"]:
                summary["by_namespace"][namespace] = 0
            summary["by_namespace"][namespace] += value
        
        return summary
    
    def get_remediation_stats(self) -> Dict[str, Any]:
        """Get statistics about remediation actions."""
        stats = {
            "total_actions": sum(
                self.remediation_duration._count.get()
            ),
            "average_duration": {},
            "total_duration": {}
        }
        
        # Calculate statistics for each action type
        for labels, count in self.remediation_duration._count.items():
            action_type = labels[0]
            total = self.remediation_duration._sum.get(labels)
            
            stats["average_duration"][action_type] = total / count
            stats["total_duration"][action_type] = total
        
        return stats 