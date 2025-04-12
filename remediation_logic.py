import pandas as pd
import numpy as np
import joblib
from utils import setup_k8s_client, safe_api_call, parse_resource_value, logger
from kubernetes import client, watch
from typing import Dict, Any, Optional, List, Tuple
import importlib.util
import time
import tensorflow.keras.models
from datetime import datetime, timedelta
import logging
import tenacity
import sys
import random
import json
import os
import traceback

# Force log flushing
logging.basicConfig(level=logging.DEBUG, force=True, stream=sys.stdout)

# Dynamically import Phase 1 modules
fetch_metrics_spec = importlib.util.spec_from_file_location("fetch_metrics", "D:\\K8\\fetch_metrics.py")
fetch_metrics = importlib.util.module_from_spec(fetch_metrics_spec)
fetch_metrics_spec.loader.exec_module(fetch_metrics)

anomaly_prediction_spec = importlib.util.spec_from_file_location("anomaly_prediction", "D:\\K8\\anomaly_prediction.py")
anomaly_prediction = importlib.util.module_from_spec(anomaly_prediction_spec)
anomaly_prediction_spec.loader.exec_module(anomaly_prediction)

# Load model, scaler, and dynamic threshold
try:
    model = tensorflow.keras.models.load_model('D:\\K8\\lstm_anomaly_model.h5')
    scaler = joblib.load('D:\\K8\\scaler.pkl')
    threshold = joblib.load('D:\\K8\\anomaly_threshold.pkl')
    logger.info("Successfully loaded lstm_anomaly_model.h5, scaler.pkl, and anomaly_threshold.pkl")
except FileNotFoundError as e:
    logger.error(f"Failed to load model, scaler, or threshold: {e}. Ensure files are in D:\\K8\\")
    raise
except Exception as e:
    logger.error(f"Error loading model, scaler, or threshold: {e}. Check file compatibility.")
    raise

# Features matching the 11-feature fetch_metrics.py and LSTM model
features = [
    'CPU Usage (%)', 'Memory Usage (%)', 'Pod Restarts',
    'Memory Usage (MB)', 'Network Receive Bytes', 'Network Transmit Bytes',
    'FS Reads Total (MB)', 'FS Writes Total (MB)',
    'Network Receive Packets Dropped (p/s)', 'Network Transmit Packets Dropped (p/s)',
    'Ready Containers'
]

sequence_length = 2  # Reduced for faster testing; revert to 10 for production

class K8sRemediation:
    def __init__(self, cooldown_period=300, max_cooldown_period=3600, confidence_threshold=0.8):
        self.base_cooldown_period = cooldown_period
        self.max_cooldown_period = max_cooldown_period
        self.remediation_history = {}
        self.deleted_pods = set()
        self.k8s_api, self.k8s_apps_api = setup_k8s_client()
        self.logger = logging.getLogger("k8s-remediation-utils")
        self.pod_history = {}
        self.confidence_threshold = confidence_threshold
        self.resource_exhaustion_threshold = {'cpu': 80.0, 'memory': 80.0}  # Thresholds for scaling
        self.max_scale_factor = 2.0  # Maximum scaling factor for resources
        self.min_scale_factor = 0.5  # Minimum scaling factor for resources
        self.max_replicas = 10  # Maximum number of replicas for scaling
        self.min_replicas = 1   # Minimum number of replicas for scaling
        self.failed_actions = {}  # Track failed actions for backoff
        self.logger.info("K8s Remediation system initialized with improved cooldown mechanism, threshold: %.4f", self.confidence_threshold)

    def _is_in_cooldown(self, resource_id: str) -> bool:
        if resource_id in self.remediation_history:
            last_action_time = self.remediation_history[resource_id]['timestamp']
            action_count = self.remediation_history[resource_id].get('action_count', 0)
            
            # Calculate exponential backoff cooldown period
            backoff_factor = min(2 ** action_count, 8)  # Cap at 8x
            jitter = random.uniform(0.8, 1.2)  # Add 20% jitter
            cooldown = min(self.base_cooldown_period * backoff_factor * jitter, self.max_cooldown_period)
            
            return (datetime.now() - last_action_time).total_seconds() < cooldown
        return False

    def _record_action(self, resource_id: str, action: str, success: bool, details: str = ""):
        current_time = datetime.now()
        
        if resource_id in self.remediation_history:
            # Increment action count if the same action was taken recently
            last_action = self.remediation_history[resource_id]
            if (current_time - last_action['timestamp']).total_seconds() < self.base_cooldown_period * 2:
                action_count = last_action.get('action_count', 0) + 1
            else:
                action_count = 1
        else:
            action_count = 1
            
        self.remediation_history[resource_id] = {
            'timestamp': current_time,
            'action': action,
            'success': success,
            'details': details,
            'action_count': action_count
        }
        
        # Track failed actions for backoff
        if not success:
            if resource_id not in self.failed_actions:
                self.failed_actions[resource_id] = []
            self.failed_actions[resource_id].append({
                'action': action,
                'timestamp': current_time,
                'details': details
            })
            # Keep only the last 5 failed actions
            self.failed_actions[resource_id] = self.failed_actions[resource_id][-5:]

    def _map_anomaly_type_to_issue(self, anomaly_type: str, status: str, metrics: Dict[str, Any]) -> Optional[str]:
        restarts = metrics.get('Pod Restarts', 0)
        cpu_usage = metrics.get('CPU Usage (%)', 0.0)
        memory_usage = metrics.get('Memory Usage (%)', 0.0)
        network_rx_dropped = metrics.get('Network Receive Packets Dropped (p/s)', 0.0)
        network_tx_dropped = metrics.get('Network Transmit Packets Dropped (p/s)', 0.0)
        fs_reads = metrics.get('FS Reads Total (MB)', 0.0)
        fs_writes = metrics.get('FS Writes Total (MB)', 0.0)
        
        self.logger.debug(f"Mapping anomaly: type={anomaly_type}, status={status}, restarts={restarts}, cpu={cpu_usage}, memory={memory_usage}")
        
        # Prioritize issues based on severity
        if restarts >= 10:
            return 'crash_loop'
        if cpu_usage > self.resource_exhaustion_threshold['cpu'] or memory_usage > self.resource_exhaustion_threshold['memory']:
            return 'resource_exhaustion'
        if network_rx_dropped > 100 or network_tx_dropped > 100:
            return 'network_issue'
        if fs_reads > 1000 or fs_writes > 1000:
            return 'io_issue'
            
        # Map based on anomaly type
        mapping = {
            'oom_kill': 'oom_kill',
            'crash_loop': 'crash_loop',
            'resource_exhaustion': 'resource_exhaustion',
            'network_issue': 'network_issue',
            'partial_failure': 'partial_failure',
            'io_issue': 'io_issue'
        }
        
        if status in ['Unknown', 'Pending']:
            return status.lower()
        return mapping.get(anomaly_type, 'unknown')

    def _fetch_pod_metrics(self, pod: client.V1Pod) -> Dict[str, Any]:
        pod_id = f"{pod.metadata.namespace}/{pod.metadata.name}"
        if pod_id in self.deleted_pods:
            self.logger.debug(f"Pod {pod_id} marked as deleted, skipping")
            return None
        
        self.logger.debug(f"Fetching metrics for {pod_id}, status: {pod.status.phase}")
        try:
            raw_metrics = fetch_metrics.fetch_metrics(pod, self.k8s_api)
            self.logger.debug(f"Raw metrics returned for {pod_id}: {raw_metrics}")
            if raw_metrics is None:
                self.logger.warning(f"No metrics fetched for {pod_id}, pod likely deleted")
                self.deleted_pods.add(pod_id)
                return None
            
            metrics = {feature: raw_metrics.get(feature, 0.0) for feature in features}
            
            if pod_id not in self.pod_history:
                self.pod_history[pod_id] = []
            self.pod_history[pod_id].append([metrics[feat] for feat in features])
            if len(self.pod_history[pod_id]) > sequence_length:
                self.pod_history[pod_id] = self.pod_history[pod_id][-sequence_length:]
            
            self.logger.debug(f"Fetched metrics for {pod_id}: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error fetching metrics for {pod_id}: {str(e)}")
            return None

    def _evaluate_remediation_effectiveness(self, resource_id: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
        if resource_id not in self.remediation_history:
            return {'success': False, 'reason': 'No remediation history'}
            
        last_action = self.remediation_history[resource_id]
        result = {'success': False, 'details': '', 'cluster_metrics': {}, 'pod_metrics': {}}
        
        # Fetch cluster-level metrics
        try:
            nodes = safe_api_call(lambda: self.k8s_api.list_node())
            pods = safe_api_call(lambda: self.k8s_api.list_pod_for_all_namespaces())
            result['cluster_metrics'] = {
                'active_nodes': len([n for n in nodes.items if n.status.conditions[-1].type == 'Ready' and n.status.conditions[-1].status == 'True']),
                'running_pods': len([p for p in pods.items if p.status.phase == 'Running']),
                'total_pods': len(pods.items)
            }
            self.logger.debug(f"Cluster metrics post-remediation for {resource_id}: {result['cluster_metrics']}")
        except Exception as e:
            self.logger.error(f"Failed to fetch cluster metrics for {resource_id}: {str(e)}")
            result['cluster_metrics'] = {'error': str(e)}

        # Check pod status and metrics
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pod = safe_api_call(lambda: self.k8s_api.read_namespaced_pod(
                    name=prediction['resource_name'],
                    namespace=prediction['namespace']
                ))
                
                # Get pod metrics
                pod_metrics = self._fetch_pod_metrics(pod)
                if pod_metrics:
                    result['pod_metrics'] = pod_metrics
                
                # Evaluate based on action type
                if last_action['action'] == 'delete_pod':
                    # For delete actions, success means the pod is gone
                    try:
                        safe_api_call(lambda: self.k8s_api.read_namespaced_pod(
                            name=prediction['resource_name'],
                            namespace=prediction['namespace']
                        ))
                        result['success'] = False
                        result['details'] = 'Pod still exists after deletion'
                    except client.rest.ApiException as e:
                        if e.status == 404:  # Not found
                            result['success'] = True
                            result['details'] = 'Pod successfully deleted'
                        else:
                            result['details'] = f'Error checking pod status: {str(e)}'
                else:
                    # For other actions, success means the pod is running
                    if pod.status.phase in ['Running', 'Succeeded']:
                        result['success'] = True
                        result['details'] = f'Pod is {pod.status.phase} post-remediation'
                        
                        # Check if metrics have improved
                        if pod_metrics:
                            cpu_usage = pod_metrics.get('CPU Usage (%)', 0.0)
                            memory_usage = pod_metrics.get('Memory Usage (%)', 0.0)
                            if cpu_usage < self.resource_exhaustion_threshold['cpu'] and \
                               memory_usage < self.resource_exhaustion_threshold['memory']:
                                result['details'] += ' and resource usage is within limits'
                            else:
                                result['details'] += ' but resource usage is still high'
                    else:
                        result['details'] = f'Pod status: {pod.status.phase}'
                
                break
            except client.rest.ApiException as e:
                self.logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    result['error'] = f'Max retries exceeded: {str(e)}'
        
        # Add historical context
        if resource_id in self.failed_actions:
            result['failed_actions'] = self.failed_actions[resource_id]
        
        return result

    def _calculate_resource_adjustment(self, current_value: str, factor: float, resource_type: str) -> str:
        """Calculate new resource value with bounds checking"""
        try:
            value, unit = parse_resource_value(current_value, return_unit=True)
            new_value = value * factor
            
            # Apply min/max bounds
            if resource_type == 'memory':
                new_value = max(min(new_value, value * self.max_scale_factor), value * self.min_scale_factor)
            elif resource_type == 'cpu':
                new_value = max(min(new_value, value * self.max_scale_factor), value * self.min_scale_factor)
            
            return f"{new_value}{unit}"
        except Exception as e:
            self.logger.error(f"Error calculating resource adjustment: {str(e)}")
            return current_value

    def _remediate_pod_oom(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        namespace, pod_name = prediction['namespace'], prediction['resource_name']
        self.logger.info(f"Remediating OOM for {pod_name} in {namespace}")
        try:
            pod = safe_api_call(lambda: self.k8s_api.read_namespaced_pod(name=pod_name, namespace=namespace))
            owner_references = pod.metadata.owner_references or []
            deployment_name = next((rs_ref.name for ref in owner_references if ref.kind == 'ReplicaSet'
                                 for rs_ref in safe_api_call(lambda: self.k8s_apps_api.read_namespaced_replica_set(
                                     name=ref.name, namespace=namespace)).metadata.owner_references
                                 if rs_ref.kind == 'Deployment'), None)
            
            if deployment_name:
                deployment = safe_api_call(lambda: self.k8s_apps_api.read_namespaced_deployment(
                    name=deployment_name, namespace=namespace))
                containers = deployment.spec.template.spec.containers or []
                
                # Calculate memory increase based on current usage
                memory_mb = float(prediction['metrics'].get('Memory Usage (MB)', 500.0))
                memory_percent = float(prediction['metrics'].get('Memory Usage (%)', 0.0))
                
                # More aggressive scaling if usage is very high
                scale_factor = 1.5
                if memory_percent > 90:
                    scale_factor = 2.0
                elif memory_percent > 95:
                    scale_factor = 2.5
                
                for i, container in enumerate(containers):
                    if not container.resources or not container.resources.limits:
                        container.resources = client.V1ResourceRequirements(limits={})
                    
                    # Set memory limits
                    if 'memory' not in container.resources.limits:
                        new_limit = f"{int(memory_mb * scale_factor / 1024)}Gi"
                        deployment.spec.template.spec.containers[i].resources.limits['memory'] = new_limit
                    else:
                        new_limit = self._calculate_resource_adjustment(
                            container.resources.limits['memory'], scale_factor, 'memory')
                        deployment.spec.template.spec.containers[i].resources.limits['memory'] = new_limit
                    
                    # Adjust memory requests proportionally
                    if container.resources.requests and 'memory' in container.resources.requests:
                        new_request = self._calculate_resource_adjustment(
                            container.resources.requests['memory'], scale_factor * 0.8, 'memory')
                        deployment.spec.template.spec.containers[i].resources.requests['memory'] = new_request
                
                # Add annotation to track remediation
                deployment.spec.template.metadata.annotations = deployment.spec.template.metadata.annotations or {}
                deployment.spec.template.metadata.annotations['lastMemoryIncrease'] = str(time.time())
                deployment.spec.template.metadata.annotations['memoryIncreaseFactor'] = str(scale_factor)
                
                safe_api_call(lambda: self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name, namespace=namespace, body=deployment), max_retries=3)
                details = f"Increased memory for {deployment_name} by factor {scale_factor}"
                self._record_action(f"pod/{namespace}/{pod_name}", "increase_memory", True, details)
                return {'action_taken': True, 'action': 'increase_resources', 'details': details}
            
            # If no deployment found, restart the pod
            safe_api_call(lambda: self.k8s_api.delete_namespaced_pod(name=pod_name, namespace=namespace))
            details = f"Restarted {pod_name} due to OOM (no deployment found)"
            self._record_action(f"pod/{namespace}/{pod_name}", "restart_pod", True, details)
            return {'action_taken': True, 'action': 'restart_pod', 'details': details}
        except client.rest.ApiException as e:
            error_msg = f"API Error remediating OOM for {pod_name}: {str(e)}"
            self.logger.error(error_msg)
            self._record_action(f"pod/{namespace}/{pod_name}", "increase_memory", False, error_msg)
            return {'action_taken': False, 'error': error_msg}

    def _remediate_pod_crash_loop(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        namespace, pod_name = prediction['namespace'], prediction['resource_name']
        self.logger.info(f"Remediating CrashLoopBackOff for {pod_name} in {namespace}")
        try:
            pod = safe_api_call(lambda: self.k8s_api.read_namespaced_pod(name=pod_name, namespace=namespace))
            container_name = pod.spec.containers[0].name
            owner_references = pod.metadata.owner_references or []
            deployment_name = next((rs_ref.name for ref in owner_references if ref.kind == 'ReplicaSet'
                                 for rs_ref in safe_api_call(lambda: self.k8s_apps_api.read_namespaced_replica_set(
                                     name=ref.name, namespace=namespace)).metadata.owner_references
                                 if rs_ref.kind == 'Deployment'), None)

            logs = None
            try:
                logs = safe_api_call(lambda: self.k8s_api.read_namespaced_pod_log(
                    name=pod_name, namespace=namespace, container=container_name, tail_lines=100))
            except (client.rest.ApiException, tenacity.RetryError) as e:
                self.logger.warning(f"Could not fetch logs for {pod_name}: {str(e)}. Proceeding without log analysis.")

            if deployment_name and prediction['metrics'].get('Pod Restarts', 0) < 10 and namespace != 'kube-system':
                deployment = safe_api_call(lambda: self.k8s_apps_api.read_namespaced_deployment(
                    name=deployment_name, namespace=namespace))
                deployment.spec.template.metadata.annotations = deployment.spec.template.metadata.annotations or {}
                deployment.spec.template.metadata.annotations['lastRestart'] = str(time.time())
                safe_api_call(lambda: self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name, namespace=namespace, body=deployment), max_retries=3)
                details = f"Triggered restart for {pod_name} via Deployment {deployment_name}"
                self._record_action(f"pod/{namespace}/{pod_name}", "restart_via_deployment", True, details)
                return {'action_taken': True, 'action': 'restart_via_deployment', 'details': details}

            if logs and ("OutOfMemoryError" in logs or "memory limit" in logs):
                if deployment_name and namespace != 'kube-system':
                    deployment = safe_api_call(lambda: self.k8s_apps_api.read_namespaced_deployment(
                        name=deployment_name, namespace=namespace))
                    containers = deployment.spec.template.spec.containers or []
                    for i, container in enumerate(containers):
                        if container.resources and container.resources.limits and 'memory' in container.resources.limits:
                            new_limit = parse_resource_value(container.resources.limits['memory'], factor=1.5)
                            deployment.spec.template.spec.containers[i].resources.limits['memory'] = new_limit
                            if 'memory' in container.resources.requests:
                                new_request = parse_resource_value(container.resources.requests['memory'], factor=1.3)
                                deployment.spec.template.spec.containers[i].resources.requests['memory'] = new_request
                    safe_api_call(lambda: self.k8s_apps_api.patch_namespaced_deployment(
                        name=deployment_name, namespace=namespace, body=deployment), max_retries=3)
                    details = f"Increased memory for {deployment_name} due to crash loop"
                    self._record_action(f"pod/{namespace}/{pod_name}", "increase_memory", True, details)
                    return {'action_taken': True, 'action': 'increase_memory', 'details': details}

            if (prediction['metrics'].get('Pod Restarts', 0) >= 10 and namespace != 'kube-system') or \
               (not deployment_name and prediction['metrics'].get('Pod Restarts', 0) >= 5):
                self.logger.warning(f"Deleting {pod_name} due to excessive restarts: {prediction['metrics'].get('Pod Restarts', 0)}")
                safe_api_call(lambda: self.k8s_api.delete_namespaced_pod(name=pod_name, namespace=namespace))
                details = f"Deleted {pod_name} due to crash loop after {prediction['metrics'].get('Pod Restarts', 0)} restarts"
                self._record_action(f"pod/{namespace}/{pod_name}", "delete_pod", True, details)
                return {'action_taken': True, 'action': 'delete_pod', 'details': details}

            return {'action_taken': False, 'reason': 'No specific action required'}
        except client.rest.ApiException as e:
            error_msg = f"API Error remediating crash loop for {pod_name}: {str(e)}"
            self.logger.error(error_msg)
            return {'action_taken': False, 'error': error_msg}

    def _remediate_resource_exhaustion(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        namespace, pod_name = prediction['namespace'], prediction['resource_name']
        self.logger.info(f"Remediating resource exhaustion for {pod_name} in {namespace}")
        try:
            pod = safe_api_call(lambda: self.k8s_api.read_namespaced_pod(name=pod_name, namespace=namespace))
            owner_references = pod.metadata.owner_references or []
            deployment_name = next((rs_ref.name for ref in owner_references if ref.kind == 'ReplicaSet'
                                 for rs_ref in safe_api_call(lambda: self.k8s_apps_api.read_namespaced_replica_set(
                                     name=ref.name, namespace=namespace)).metadata.owner_references
                                 if rs_ref.kind == 'Deployment'), None)
            
            if deployment_name and namespace != 'kube-system':
                deployment = safe_api_call(lambda: self.k8s_apps_api.read_namespaced_deployment(
                    name=deployment_name, namespace=namespace))
                
                # Get current resource usage
                cpu_usage = float(prediction['metrics'].get('CPU Usage (%)', 0.0))
                memory_usage = float(prediction['metrics'].get('Memory Usage (%)', 0.0))
                
                # Get predicted resource usage if available
                predicted_cpu = prediction.get('predicted_metrics', {}).get('CPU Usage (%)', cpu_usage)
                predicted_memory = prediction.get('predicted_metrics', {}).get('Memory Usage (%)', memory_usage)
                
                # Use the higher of current or predicted values for proactive scaling
                cpu_usage = max(cpu_usage, predicted_cpu)
                memory_usage = max(memory_usage, predicted_memory)
                
                # Determine scaling strategy
                scale_replicas = False
                scale_resources = False
                scale_factor = 1.0
                
                # Calculate severity of resource exhaustion
                cpu_severity = max(0, (cpu_usage - self.resource_exhaustion_threshold['cpu']) / 100)
                memory_severity = max(0, (memory_usage - self.resource_exhaustion_threshold['memory']) / 100)
                
                # Determine scaling strategy based on severity
                if cpu_usage > self.resource_exhaustion_threshold['cpu']:
                    scale_replicas = True
                    # Calculate scale factor based on severity
                    scale_factor = 1.0 + (cpu_severity * 0.5)  # 50% increase per severity unit
                
                if memory_usage > self.resource_exhaustion_threshold['memory']:
                    scale_resources = True
                    # Calculate memory scale factor based on severity
                    memory_scale_factor = 1.0 + (memory_severity * 0.5)  # 50% increase per severity unit
                
                # If both are high, prioritize scaling replicas for CPU-bound workloads
                if cpu_usage > 90 and memory_usage > 90:
                    scale_replicas = True
                    scale_resources = False
                    # More aggressive scaling for critical situations
                    scale_factor = 1.0 + (max(cpu_severity, memory_severity) * 0.75)
                
                # Apply scaling strategy
                if scale_replicas:
                    current_replicas = deployment.spec.replicas or 1
                    # Calculate new replicas based on usage and severity
                    new_replicas = min(int(current_replicas * scale_factor), self.max_replicas)
                    
                    if new_replicas > current_replicas:
                        # Check if HPA exists and update it instead
                        try:
                            hpa = safe_api_call(lambda: self.k8s_api.read_namespaced_horizontal_pod_autoscaler(
                                name=deployment_name, namespace=namespace))
                            if hpa:
                                # Update HPA min replicas if needed
                                if hpa.spec.min_replicas is None or hpa.spec.min_replicas < new_replicas:
                                    hpa.spec.min_replicas = new_replicas
                                    safe_api_call(lambda: self.k8s_api.patch_namespaced_horizontal_pod_autoscaler(
                                        name=deployment_name, namespace=namespace, body=hpa), max_retries=3)
                                    details = f"Updated HPA min replicas for {deployment_name} to {new_replicas}"
                                    self._record_action(f"pod/{namespace}/{pod_name}", "update_hpa", True, details)
                                    return {'action_taken': True, 'action': 'update_hpa', 'details': details}
                        except client.rest.ApiException as e:
                            if e.status != 404:  # Not found is expected if HPA doesn't exist
                                self.logger.warning(f"Error checking HPA for {deployment_name}: {str(e)}")
                        
                        # If no HPA or update failed, update deployment directly
                        deployment.spec.replicas = new_replicas
                        safe_api_call(lambda: self.k8s_apps_api.patch_namespaced_deployment(
                            name=deployment_name, namespace=namespace, body=deployment), max_retries=3)
                        details = f"Scaled {deployment_name} from {current_replicas} to {new_replicas} replicas due to CPU exhaustion"
                        self._record_action(f"pod/{namespace}/{pod_name}", "scale_deployment", True, details)
                        return {'action_taken': True, 'action': 'scale_deployment', 'details': details}
                
                if scale_resources:
                    containers = deployment.spec.template.spec.containers or []
                    for i, container in enumerate(containers):
                        if container.resources and container.resources.limits:
                            # Scale CPU limits
                            if 'cpu' in container.resources.limits:
                                new_cpu_limit = self._calculate_resource_adjustment(
                                    container.resources.limits['cpu'], memory_scale_factor, 'cpu')
                                deployment.spec.template.spec.containers[i].resources.limits['cpu'] = new_cpu_limit
                            
                            # Scale memory limits
                            if 'memory' in container.resources.limits:
                                new_memory_limit = self._calculate_resource_adjustment(
                                    container.resources.limits['memory'], memory_scale_factor, 'memory')
                                deployment.spec.template.spec.containers[i].resources.limits['memory'] = new_memory_limit
                            
                            # Also update requests to maintain ratio
                            if container.resources.requests:
                                if 'cpu' in container.resources.requests and 'cpu' in container.resources.limits:
                                    cpu_limit = parse_resource_value(container.resources.limits['cpu'])
                                    cpu_request = parse_resource_value(container.resources.requests['cpu'])
                                    ratio = cpu_request / cpu_limit if cpu_limit > 0 else 0.5
                                    new_cpu_request = parse_resource_value(new_cpu_limit, factor=ratio)
                                    deployment.spec.template.spec.containers[i].resources.requests['cpu'] = new_cpu_request
                                
                                if 'memory' in container.resources.requests and 'memory' in container.resources.limits:
                                    memory_limit = parse_resource_value(container.resources.limits['memory'])
                                    memory_request = parse_resource_value(container.resources.requests['memory'])
                                    ratio = memory_request / memory_limit if memory_limit > 0 else 0.5
                                    new_memory_request = parse_resource_value(new_memory_limit, factor=ratio)
                                    deployment.spec.template.spec.containers[i].resources.requests['memory'] = new_memory_request
                    
                    # Add annotation to track remediation
                    deployment.spec.template.metadata.annotations = deployment.spec.template.metadata.annotations or {}
                    deployment.spec.template.metadata.annotations['lastResourceIncrease'] = str(time.time())
                    deployment.spec.template.metadata.annotations['resourceIncreaseFactor'] = str(memory_scale_factor)
                    
                    # Create VPA recommendation annotation
                    vpa_recommendation = {
                        'cpu': {
                            'request': new_cpu_request if 'new_cpu_request' in locals() else None,
                            'limit': new_cpu_limit if 'new_cpu_limit' in locals() else None
                        },
                        'memory': {
                            'request': new_memory_request if 'new_memory_request' in locals() else None,
                            'limit': new_memory_limit if 'new_memory_limit' in locals() else None
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    deployment.spec.template.metadata.annotations['vpa-recommendation'] = str(vpa_recommendation)
                    
                    safe_api_call(lambda: self.k8s_apps_api.patch_namespaced_deployment(
                        name=deployment_name, namespace=namespace, body=deployment), max_retries=3)
                    details = f"Increased resources for {deployment_name} due to resource exhaustion"
                    self._record_action(f"pod/{namespace}/{pod_name}", "increase_resources", True, details)
                    return {'action_taken': True, 'action': 'increase_resources', 'details': details}
            
            # Fallback: Restart pod if no deployment or scaling not applicable
            safe_api_call(lambda: self.k8s_api.delete_namespaced_pod(name=pod_name, namespace=namespace))
            details = f"Restarted {pod_name} due to resource exhaustion (no deployment found or scaling not applicable)"
            self._record_action(f"pod/{namespace}/{pod_name}", "restart_pod", True, details)
            return {'action_taken': True, 'action': 'restart_pod', 'details': details}
        except client.rest.ApiException as e:
            error_msg = f"API Error remediating resource exhaustion for {pod_name}: {str(e)}"
            self.logger.error(error_msg)
            self._record_action(f"pod/{namespace}/{pod_name}", "scale_deployment", False, error_msg)
            return {'action_taken': False, 'error': error_msg}

    def _remediate_pod_unknown_state(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        namespace, pod_name = prediction['namespace'], prediction['resource_name']
        self.logger.info(f"Remediating Unknown state for {pod_name} in {namespace}")
        try:
            delete_options = client.V1DeleteOptions(grace_period_seconds=0)
            safe_api_call(lambda: self.k8s_api.delete_namespaced_pod(
                name=pod_name, namespace=namespace, body=delete_options))
            details = f"Force deleted {pod_name} in Unknown state"
            self._record_action(f"pod/{namespace}/{pod_name}", "force_delete", True, details)
            return {'action_taken': True, 'action': 'force_delete', 'details': details}
        except Exception as e:
            error_msg = f"Error remediating unknown state for {pod_name}: {str(e)}"
            self.logger.error(error_msg)
            return {'action_taken': False, 'error': error_msg}

    def remediate_pending_pod(self, namespace: str, pod_name: str) -> Dict[str, Any]:
        self.logger.info(f"Remediating Pending state for {pod_name} in {namespace}")
        try:
            pod = safe_api_call(lambda: self.k8s_api.read_namespaced_pod(name=pod_name, namespace=namespace))
            if pod.status.conditions:
                for condition in pod.status.conditions:
                    if condition.reason == "Unschedulable" and "insufficient" in condition.message.lower():
                        if "memory" in condition.message.lower():
                            return self._reduce_pod_memory_requests(namespace, pod_name)
                        elif "cpu" in condition.message.lower():
                            return self._reduce_pod_cpu_requests(namespace, pod_name)
            owner_references = pod.metadata.owner_references or []
            is_part_of_controller = any(ref.kind in ['ReplicaSet', 'StatefulSet', 'DaemonSet'] for ref in owner_references)
            if is_part_of_controller:
                safe_api_call(lambda: self.k8s_api.delete_namespaced_pod(name=pod_name, namespace=namespace))
                details = f"Deleted {pod_name} to trigger recreation"
                self._record_action(f"pod/{namespace}/{pod_name}", "delete_pending", True, details)
                return {'action_taken': True, 'action': 'delete_pending', 'details': details}
            details = f"No action for {pod_name} as not managed by controller"
            return {'action_taken': False, 'action': 'no_action', 'details': details}
        except Exception as e:
            error_msg = f"Error remediating pending pod {pod_name}: {str(e)}"
            self.logger.error(error_msg)
            return {'action_taken': False, 'error': error_msg}

    def _reduce_pod_memory_requests(self, namespace: str, pod_name: str) -> Dict[str, Any]:
        try:
            pod = safe_api_call(lambda: self.k8s_api.read_namespaced_pod(name=pod_name, namespace=namespace))
            deployment_name = next((rs_ref.name for ref in pod.metadata.owner_references or [] if ref.kind == 'ReplicaSet'
                                 for rs_ref in safe_api_call(lambda: self.k8s_apps_api.read_namespaced_replica_set(
                                     name=ref.name, namespace=namespace)).metadata.owner_references
                                 if rs_ref.kind == 'Deployment'), None)
            if deployment_name:
                deployment = safe_api_call(lambda: self.k8s_apps_api.read_namespaced_deployment(
                    name=deployment_name, namespace=namespace))
                containers = deployment.spec.template.spec.containers or []
                for i, container in enumerate(containers):
                    if container.resources and container.resources.requests and 'memory' in container.resources.requests:
                        new_request = parse_resource_value(container.resources.requests['memory'], factor=0.8)
                        deployment.spec.template.spec.containers[i].resources.requests['memory'] = new_request
                safe_api_call(lambda: self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name, namespace=namespace, body=deployment), max_retries=3)
                details = f"Reduced memory requests for {deployment_name}"
                self._record_action(f"pod/{namespace}/{pod_name}", "reduce_memory_requests", True, details)
                return {'action_taken': True, 'action': 'reduce_memory_requests', 'details': details}
            details = f"No deployment for {pod_name}, cannot adjust memory"
            return {'action_taken': False, 'action': 'no_action', 'details': details}
        except Exception as e:
            error_msg = f"Error reducing memory for {pod_name}: {str(e)}"
            self.logger.error(error_msg)
            return {'action_taken': False, 'error': error_msg}

    def _reduce_pod_cpu_requests(self, namespace: str, pod_name: str) -> Dict[str, Any]:
        try:
            pod = safe_api_call(lambda: self.k8s_api.read_namespaced_pod(name=pod_name, namespace=namespace))
            deployment_name = next((rs_ref.name for ref in pod.metadata.owner_references or [] if ref.kind == 'ReplicaSet'
                                 for rs_ref in safe_api_call(lambda: self.k8s_apps_api.read_namespaced_replica_set(
                                     name=ref.name, namespace=namespace)).metadata.owner_references
                                 if rs_ref.kind == 'Deployment'), None)
            if deployment_name:
                deployment = safe_api_call(lambda: self.k8s_apps_api.read_namespaced_deployment(
                    name=deployment_name, namespace=namespace))
                containers = deployment.spec.template.spec.containers or []
                for i, container in enumerate(containers):
                    if container.resources and container.resources.requests and 'cpu' in container.resources.requests:
                        new_request = parse_resource_value(container.resources.requests['cpu'], factor=0.8)
                        deployment.spec.template.spec.containers[i].resources.requests['cpu'] = new_request
                safe_api_call(lambda: self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name, namespace=namespace, body=deployment), max_retries=3)
                details = f"Reduced CPU requests for {deployment_name}"
                self._record_action(f"pod/{namespace}/{pod_name}", "reduce_cpu_requests", True, details)
                return {'action_taken': True, 'action': 'reduce_cpu_requests', 'details': details}
            details = f"No deployment for {pod_name}, cannot adjust CPU"
            return {'action_taken': False, 'action': 'no_action', 'details': details}
        except Exception as e:
            error_msg = f"Error reducing CPU for {pod_name}: {str(e)}"
            self.logger.error(error_msg)
            return {'action_taken': False, 'error': error_msg}

    def _remediate_network_issue(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        namespace, pod_name = prediction['namespace'], prediction['resource_name']
        self.logger.info(f"Remediating network issue for {pod_name} in {namespace}")
        try:
            safe_api_call(lambda: self.k8s_api.delete_namespaced_pod(name=pod_name, namespace=namespace))
            details = f"Restarted {pod_name} to resolve network issue"
            self._record_action(f"pod/{namespace}/{pod_name}", "restart_pod", True, details)
            return {'action_taken': True, 'action': 'restart_pod', 'details': details}
        except Exception as e:
            error_msg = f"Error remediating network issue for {pod_name}: {str(e)}"
            self.logger.error(error_msg)
            return {'action_taken': False, 'error': error_msg}

    def _remediate_partial_failure(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        namespace, pod_name = prediction['namespace'], prediction['resource_name']
        self.logger.info(f"Remediating partial failure for {pod_name} in {namespace}")
        try:
            safe_api_call(lambda: self.k8s_api.delete_namespaced_pod(name=pod_name, namespace=namespace))
            details = f"Restarted {pod_name} due to partial container failure"
            self._record_action(f"pod/{namespace}/{pod_name}", "restart_pod", True, details)
            return {'action_taken': True, 'action': 'restart_pod', 'details': details}
        except Exception as e:
            error_msg = f"Error remediating partial failure for {pod_name}: {str(e)}"
            self.logger.error(error_msg)
            return {'action_taken': False, 'error': error_msg}

    def _remediate_io_issue(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        namespace, pod_name = prediction['namespace'], prediction['resource_name']
        self.logger.info(f"Remediating I/O issue for {pod_name} in {namespace}")
        try:
            safe_api_call(lambda: self.k8s_api.delete_namespaced_pod(name=pod_name, namespace=namespace))
            details = f"Restarted {pod_name} to resolve I/O issue"
            self._record_action(f"pod/{namespace}/{pod_name}", "restart_pod", True, details)
            return {'action_taken': True, 'action': 'restart_pod', 'details': details}
        except Exception as e:
            error_msg = f"Error remediating I/O issue for {pod_name}: {str(e)}"
            self.logger.error(error_msg)
            return {'action_taken': False, 'error': error_msg}

    def _scale_deployment(self, namespace: str, deployment_name: str, scale_factor: float, reason: str) -> Dict[str, Any]:
        """Scale a deployment based on the provided scale factor and reason."""
        try:
            deployment = safe_api_call(lambda: self.k8s_apps_api.read_namespaced_deployment(
                name=deployment_name, namespace=namespace))
            
            current_replicas = deployment.spec.replicas or 1
            new_replicas = min(max(int(current_replicas * scale_factor), self.min_replicas), self.max_replicas)
            
            if new_replicas == current_replicas:
                return {'action_taken': False, 'reason': 'No scaling needed'}
            
            # Update deployment
            deployment.spec.replicas = new_replicas
            
            # Add scaling annotation
            deployment.spec.template.metadata.annotations = deployment.spec.template.metadata.annotations or {}
            deployment.spec.template.metadata.annotations['lastScaling'] = str(time.time())
            deployment.spec.template.metadata.annotations['scalingFactor'] = str(scale_factor)
            deployment.spec.template.metadata.annotations['scalingReason'] = reason
            
            safe_api_call(lambda: self.k8s_apps_api.patch_namespaced_deployment(
                name=deployment_name, namespace=namespace, body=deployment), max_retries=3)
            
            details = f"Scaled {deployment_name} from {current_replicas} to {new_replicas} replicas due to {reason}"
            self._record_action(f"deployment/{namespace}/{deployment_name}", "scale_deployment", True, details)
            return {'action_taken': True, 'action': 'scale_deployment', 'details': details}
            
        except Exception as e:
            error_msg = f"Error scaling deployment {deployment_name}: {str(e)}"
            self.logger.error(error_msg)
            self._record_action(f"deployment/{namespace}/{deployment_name}", "scale_deployment", False, error_msg)
            return {'action_taken': False, 'error': error_msg}

    def _determine_scaling_action(self, prediction: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Determine if and how to scale based on anomaly metrics."""
        metrics = prediction.get('metrics', {})
        predicted_metrics = prediction.get('predicted_metrics', {})
        
        # Get current and predicted resource usage
        cpu_usage = max(float(metrics.get('CPU Usage (%)', 0.0)), 
                       float(predicted_metrics.get('CPU Usage (%)', 0.0)))
        memory_usage = max(float(metrics.get('Memory Usage (%)', 0.0)), 
                          float(predicted_metrics.get('Memory Usage (%)', 0.0)))
        
        # Calculate severity
        cpu_severity = max(0, (cpu_usage - self.resource_exhaustion_threshold['cpu']) / 100)
        memory_severity = max(0, (memory_usage - self.resource_exhaustion_threshold['memory']) / 100)
        
        # Determine scaling factor based on severity
        if cpu_severity > 0.5 or memory_severity > 0.5:
            return True, 2.0, "severe resource exhaustion"
        elif cpu_severity > 0.3 or memory_severity > 0.3:
            return True, 1.5, "moderate resource exhaustion"
        elif cpu_severity > 0.1 or memory_severity > 0.1:
            return True, 1.2, "mild resource exhaustion"
        
        return False, 1.0, ""

    def remediate_issue(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Main remediation method that determines and executes the appropriate remediation action."""
        if prediction['resource_type'] == 'pod':
            resource_id = f"pod/{prediction['namespace']}/{prediction['resource_name']}"
            
            if self._is_in_cooldown(resource_id):
                self.logger.info(f"Resource {resource_id} is in cooldown period")
                return {'action_taken': False, 'reason': 'In cooldown period'}
            
            issue_type = prediction.get('issue_type', 'unknown')
            self.logger.info(f"Remediating {issue_type} for {resource_id}")
            
            # First, check if scaling is needed
            should_scale, scale_factor, scale_reason = self._determine_scaling_action(prediction)
            
            if should_scale:
                # Get deployment name if available
                pod = safe_api_call(lambda: self.k8s_api.read_namespaced_pod(
                    name=prediction['resource_name'], 
                    namespace=prediction['namespace']))
                
                deployment_name = next((rs_ref.name for ref in pod.metadata.owner_references or [] 
                                     if ref.kind == 'ReplicaSet'
                                     for rs_ref in safe_api_call(lambda: self.k8s_apps_api.read_namespaced_replica_set(
                                         name=ref.name, 
                                         namespace=prediction['namespace'])).metadata.owner_references
                                     if rs_ref.kind == 'Deployment'), None)
                
                if deployment_name:
                    scale_result = self._scale_deployment(
                        prediction['namespace'],
                        deployment_name,
                        scale_factor,
                        scale_reason
                    )
                    if scale_result['action_taken']:
                        return scale_result
            
            # Continue with other remediation actions if scaling wasn't possible or sufficient
            remediation_map = {
                'oom_kill': self._remediate_pod_oom,
                'crash_loop': self._remediate_pod_crash_loop,
                'resource_exhaustion': self._remediate_resource_exhaustion,
                'network_issue': self._remediate_network_issue,
                'partial_failure': self._remediate_partial_failure,
                'io_issue': self._remediate_io_issue,
                'unknown': self._remediate_pod_unknown_state,
                'pending': self.remediate_pending_pod
            }
            
            if issue_type in remediation_map:
                try:
                    result = remediation_map[issue_type](prediction)
                    
                    # If remediation failed, try with adjusted thresholds
                    if not result.get('action_taken', False) and issue_type == 'resource_exhaustion':
                        # Try remediation again with higher thresholds
                        original_threshold = self.resource_exhaustion_threshold.copy()
                        self.resource_exhaustion_threshold = {
                            'cpu': original_threshold['cpu'] * 0.9,  # Lower threshold by 10%
                            'memory': original_threshold['memory'] * 0.9
                        }
                        
                        result = remediation_map[issue_type](prediction)
                        
                        # Restore original thresholds
                        self.resource_exhaustion_threshold = original_threshold
                
                    return result
                except Exception as e:
                    error_msg = f"Error during remediation for {resource_id}: {str(e)}"
                    self.logger.error(error_msg)
                    self._record_action(resource_id, "remediation_error", False, error_msg)
                    return {'action_taken': False, 'error': error_msg}
            else:
                self.logger.warning(f"No remediation for {issue_type}")
                return {'action_taken': False, 'reason': 'No remediation defined'}
        else:
            self.logger.warning(f"Unsupported resource type: {prediction['resource_type']}")
            return {'action_taken': False, 'reason': 'Unsupported resource type'}

    def monitor_cluster(self, interval=10):  # Reduced interval for faster testing
        self.logger.info("Starting real-time cluster monitoring")
        w = watch.Watch()
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while True:
                try:
                    for event in w.stream(self.k8s_api.list_pod_for_all_namespaces, timeout_seconds=interval):
                        pod = event['object']
                        pod_id = f"{pod.metadata.namespace}/{pod.metadata.name}"
                        self.logger.debug(f"Processing event for {pod_id}, status: {pod.status.phase}, event type: {event['type']}")
                        
                        try:
                            metrics = self._fetch_pod_metrics(pod)
                            if metrics is None:
                                self.logger.debug(f"Skipping {pod_id} due to no metrics")
                                continue
                            
                            self.logger.debug(f"Pod history length for {pod_id}: {len(self.pod_history[pod_id])}")
                            if len(self.pod_history[pod_id]) == sequence_length:
                                metrics_seq = pd.DataFrame(self.pod_history[pod_id], columns=features)
                                self.logger.debug(f"Metrics sequence for {pod_id}: {metrics_seq.to_dict()}")
                                
                                # Check if pod is in a problematic state
                                if pod.status.phase in ['Pending', 'Unknown'] or pod.status.phase is None:
                                    prediction = {
                                        'resource_type': 'pod',
                                        'resource_name': pod.metadata.name,
                                        'namespace': pod.metadata.namespace,
                                        'issue_type': pod.status.phase.lower() if pod.status.phase else 'unknown',
                                        'confidence': 1.0,  # High confidence for direct status issues
                                        'metrics': metrics
                                    }
                                    result = self.remediate_issue(prediction)
                                    self.logger.info(f"Processed {pod_id} with direct status issue: {result}")
                                    continue
                                
                                # Check for anomalies
                                prediction_df = anomaly_prediction.predict_anomalies(metrics_seq, sequence_length)
                                self.logger.debug(f"Prediction for {pod_id}: {prediction_df.to_dict()}")
                                
                                if not prediction_df.empty and prediction_df['predicted_anomaly'].iloc[0] == 1:
                                    confidence = prediction_df['anomaly_probability'].iloc[0]
                                    
                                    # Only proceed if confidence is above threshold
                                    if confidence >= self.confidence_threshold:
                                        prediction = {
                                            'resource_type': 'pod',
                                            'resource_name': pod.metadata.name,
                                            'namespace': pod.metadata.namespace,
                                            'issue_type': self._map_anomaly_type_to_issue(
                                                prediction_df['anomaly_type'].iloc[0], pod.status.phase, metrics),
                                            'confidence': confidence,
                                            'metrics': metrics
                                        }
                                        result = self.remediate_issue(prediction)
                                        self.logger.info(f"Processed {pod_id}: {result}")
                                    else:
                                        self.logger.info(f"Skipped {pod_id} due to low confidence: {confidence}")
                                else:
                                    self.logger.info(f"Processed {pod_id}: {{'action_taken': False, 'details': 'No anomaly detected'}}")
                            else:
                                self.logger.debug(f"Waiting for {sequence_length} samples for {pod_id}, current: {len(self.pod_history[pod_id])}")
                            
                            # Reset consecutive errors on successful processing
                            consecutive_errors = 0
                            
                        except Exception as e:
                            self.logger.error(f"Error processing event for {pod_id}: {str(e)}")
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                self.logger.error(f"Too many consecutive errors ({consecutive_errors}), restarting watch")
                                w.stop()
                                time.sleep(5)  # Wait before restarting
                                w = watch.Watch()
                                consecutive_errors = 0
                            continue
                            
                except Exception as e:
                    self.logger.error(f"Watch stream error: {str(e)}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error(f"Too many consecutive watch errors ({consecutive_errors}), restarting watch")
                        w.stop()
                        time.sleep(5)  # Wait before restarting
                        w = watch.Watch()
                        consecutive_errors = 0
                    continue
                    
        except Exception as e:
            self.logger.error(f"Monitoring error: {str(e)}")
            raise
        finally:
            w.stop()
            self.logger.info("Stopped real-time cluster monitoring")

    def process_expert_dataset(self, data_source):
        import json
        try:
            if isinstance(data_source, str):
                data = json.loads(data_source)
            elif hasattr(data_source, 'read'):
                data = json.load(data_source)
            for entry in data:
                metrics = entry.get('metrics', {})
                for feature in features:
                    if feature not in metrics:
                        metrics[feature] = 0.0
                metrics_df = pd.DataFrame([metrics], columns=features)
                prediction_df = anomaly_prediction.predict_anomalies(metrics_df, sequence_length)
                if not prediction_df.empty and prediction_df['predicted_anomaly'].iloc[0] == 1:
                    prediction = {
                        'resource_type': entry.get('resource_type', 'pod'),
                        'resource_name': entry.get('name', entry.get('pod_name')),
                        'namespace': entry.get('namespace', 'default'),
                        'issue_type': self._map_anomaly_type_to_issue(
                            prediction_df['anomaly_type'].iloc[0], entry.get('status', 'Running'), metrics),
                        'confidence': prediction_df['anomaly_probability'].iloc[0],
                        'metrics': metrics
                    }
                    result = self.remediate_issue(prediction)
                    self.logger.info(f"Remediation for {prediction['resource_name']}: {result}")
        except Exception as e:
            self.logger.error(f"Error processing expert dataset: {str(e)}")

def parse_resource_value(resource_str: str) -> float:
    """
    Parse a Kubernetes resource value string into a float.
    
    Args:
        resource_str: Resource value string (e.g., '100m', '1Gi', '0.5')
        
    Returns:
        Float value representing the resource quantity
    """
    if not resource_str:
        return 0.0
    
    # Handle CPU values (cores or millicores)
    if resource_str.endswith('m'):
        return float(resource_str[:-1]) / 1000.0
    
    # Handle memory values
    multipliers = {
        'Ki': 1024,
        'Mi': 1024**2,
        'Gi': 1024**3,
        'Ti': 1024**4,
        'Pi': 1024**5,
        'Ei': 1024**6
    }
    
    for suffix, multiplier in multipliers.items():
        if resource_str.endswith(suffix):
            return float(resource_str[:-len(suffix)]) * multiplier
    
    # Handle plain numbers
    try:
        return float(resource_str)
    except ValueError:
        return 0.0

def format_resource_value(value: float, resource_type: str) -> str:
    """
    Format a float value into a Kubernetes resource string.
    
    Args:
        value: Float value to format
        resource_type: Type of resource ('cpu' or 'memory')
        
    Returns:
        Formatted resource string
    """
    if resource_type == 'cpu':
        # Format CPU values in cores or millicores
        if value < 0.1:
            return f"{int(value * 1000)}m"
        return f"{value:.3f}"
    
    elif resource_type == 'memory':
        # Format memory values in Mi, Gi, etc.
        if value < 1024**2:  # Less than 1 Mi
            return f"{int(value / 1024)}Ki"
        elif value < 1024**3:  # Less than 1 Gi
            return f"{int(value / 1024**2)}Mi"
        elif value < 1024**4:  # Less than 1 Ti
            return f"{value / 1024**3:.2f}Gi"
        else:
            return f"{value / 1024**3:.2f}Gi"
    
    return str(value)

def calculate_resource_recommendations(historical_usage: List[Dict[str, float]], 
                                      percentile: float = 95.0,
                                      buffer_factor: float = 1.2) -> Dict[str, Dict[str, float]]:
    """
    Calculate recommended resource requests and limits based on historical usage.
    
    Args:
        historical_usage: List of dictionaries containing historical resource usage data
        percentile: Percentile to use for calculating limits (default: 95.0)
        buffer_factor: Factor to add as a buffer to the calculated values (default: 1.2)
        
    Returns:
        Dictionary with recommended requests and limits for CPU and memory
    """
    if not historical_usage:
        return {
            'requests': {'cpu': 0.0, 'memory': 0.0},
            'limits': {'cpu': 0.0, 'memory': 0.0}
        }
    
    # Extract CPU and memory usage values
    cpu_usage = [entry.get('cpu_usage', 0.0) for entry in historical_usage]
    memory_usage = [entry.get('memory_usage', 0.0) for entry in historical_usage]
    
    # Calculate average for requests (50th percentile)
    cpu_request = np.percentile(cpu_usage, 50) * buffer_factor
    memory_request = np.percentile(memory_usage, 50) * buffer_factor
    
    # Calculate percentile for limits
    cpu_limit = np.percentile(cpu_usage, percentile) * buffer_factor
    memory_limit = np.percentile(memory_usage, percentile) * buffer_factor
    
    return {
        'requests': {
            'cpu': cpu_request,
            'memory': memory_request
        },
        'limits': {
            'cpu': cpu_limit,
            'memory': memory_limit
        }
    }

def generate_remediation_recommendations(
    anomaly_results: Dict[str, Any],
    historical_usage: List[Dict[str, float]],
    current_resources: Dict[str, Dict[str, str]],
    namespace: str,
    pod_name: str
) -> Dict[str, Any]:
    """
    Generate remediation recommendations based on anomaly detection results and historical usage.
    
    Args:
        anomaly_results: Dictionary containing anomaly detection results
        historical_usage: List of dictionaries containing historical resource usage data
        current_resources: Dictionary containing current resource requests and limits
        namespace: Kubernetes namespace
        pod_name: Name of the pod
        
    Returns:
        Dictionary containing remediation recommendations
    """
    recommendations = {
        'namespace': namespace,
        'pod_name': pod_name,
        'timestamp': datetime.now().isoformat(),
        'anomaly_score': anomaly_results.get('anomaly_score', 0.0),
        'is_anomaly': anomaly_results.get('is_anomaly', False),
        'recommendations': {}
    }
    
    # Calculate recommended resources
    recommended_resources = calculate_resource_recommendations(historical_usage)
    
    # Parse current resources
    current_requests = {
        'cpu': parse_resource_value(current_resources.get('requests', {}).get('cpu', '0')),
        'memory': parse_resource_value(current_resources.get('requests', {}).get('memory', '0'))
    }
    current_limits = {
        'cpu': parse_resource_value(current_resources.get('limits', {}).get('cpu', '0')),
        'memory': parse_resource_value(current_resources.get('limits', {}).get('memory', '0'))
    }
    
    # Generate recommendations for each resource type
    for resource_type in ['cpu', 'memory']:
        current_request = current_requests[resource_type]
        current_limit = current_limits[resource_type]
        recommended_request = recommended_resources['requests'][resource_type]
        recommended_limit = recommended_resources['limits'][resource_type]
        
        # Calculate percentage differences
        request_diff_pct = ((recommended_request - current_request) / current_request * 100) if current_request > 0 else float('inf')
        limit_diff_pct = ((recommended_limit - current_limit) / current_limit * 100) if current_limit > 0 else float('inf')
        
        recommendations['recommendations'][resource_type] = {
            'current': {
                'request': format_resource_value(current_request, resource_type),
                'limit': format_resource_value(current_limit, resource_type)
            },
            'recommended': {
                'request': format_resource_value(recommended_request, resource_type),
                'limit': format_resource_value(recommended_limit, resource_type)
            },
            'difference_percentage': {
                'request': request_diff_pct,
                'limit': limit_diff_pct
            },
            'action_required': abs(request_diff_pct) > 20 or abs(limit_diff_pct) > 20
        }
    
    return recommendations