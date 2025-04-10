import pandas as pd
import numpy as np
import joblib
from utils import setup_k8s_client, safe_api_call, parse_resource_value, logger
from kubernetes import client, watch
from typing import Dict, Any, Optional
import importlib.util
import time
import tensorflow.keras.models
from datetime import datetime
import logging
import tenacity
import sys

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
    def __init__(self, cooldown_period=300):
        self.cooldown_period = cooldown_period
        self.remediation_history = {}
        self.deleted_pods = set()
        self.k8s_api, self.k8s_apps_api = setup_k8s_client()
        self.logger = logging.getLogger("k8s-remediation-utils")
        self.pod_history = {}
        self.threshold = 0.8  # Temporary lower threshold for testing
        self.resource_exhaustion_threshold = {'cpu': 80.0, 'memory': 80.0}  # Thresholds for scaling
        self.logger.info("K8s Remediation system initialized with Phase 1 integration, threshold: %.4f", self.threshold)

    def _is_in_cooldown(self, resource_id: str) -> bool:
        if resource_id in self.remediation_history:
            last_action_time = self.remediation_history[resource_id]['timestamp']
            return (datetime.now() - last_action_time).total_seconds() < self.cooldown_period
        return False

    def _record_action(self, resource_id: str, action: str, success: bool, details: str = ""):
        self.remediation_history[resource_id] = {
            'timestamp': datetime.now(),
            'action': action,
            'success': success,
            'details': details
        }

    def _map_anomaly_type_to_issue(self, anomaly_type: str, status: str, metrics: Dict[str, Any]) -> Optional[str]:
        restarts = metrics.get('Pod Restarts', 0)
        cpu_usage = metrics.get('CPU Usage (%)', 0.0)
        memory_usage = metrics.get('Memory Usage (%)', 0.0)
        self.logger.debug(f"Mapping anomaly: type={anomaly_type}, status={status}, restarts={restarts}, cpu={cpu_usage}, memory={memory_usage}")
        if restarts >= 10:
            return 'crash_loop'
        if cpu_usage > self.resource_exhaustion_threshold['cpu'] or memory_usage > self.resource_exhaustion_threshold['memory']:
            return 'resource_exhaustion'
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
        result = {'success': False, 'details': '', 'cluster_metrics': {}}
        
        # Fetch cluster-level metrics (simplified via node/pod counts for now)
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

        if last_action['action'] == 'delete_pod' and last_action['success']:
            result['success'] = True
            result['details'] = 'Pod successfully deleted as intended'
        elif last_action['success']:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    pod = safe_api_call(lambda: self.k8s_api.read_namespaced_pod(
                        name=prediction['resource_name'],
                        namespace=prediction['namespace']
                    ))
                    if pod.status.phase in ['Running', 'Succeeded']:
                        result['success'] = True
                        result['details'] = 'Pod is running post-remediation'
                        break
                    result['details'] = f'Pod status: {pod.status.phase}'
                    break
                except client.rest.ApiException as e:
                    self.logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            if not result['success']:
                result['error'] = 'Max retries exceeded or pod not running'
        else:
            result['reason'] = 'Remediation failed initially'
        
        return result

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
                for i, container in enumerate(containers):
                    if not container.resources or not container.resources.limits:
                        container.resources = client.V1ResourceRequirements(limits={})
                    if 'memory' not in container.resources.limits:
                        memory_mb = float(prediction['metrics'].get('Memory Usage (MB)', 500.0))
                        new_limit = f"{int(memory_mb * 1.5 / 1024)}Gi"
                        deployment.spec.template.spec.containers[i].resources.limits['memory'] = new_limit
                        if container.resources.requests and 'memory' in container.resources.requests:
                            new_request = parse_resource_value(container.resources.requests['memory'], factor=1.3)
                            deployment.spec.template.spec.containers[i].resources.requests['memory'] = new_request
                safe_api_call(lambda: self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name, namespace=namespace, body=deployment), max_retries=3)
                details = f"Increased memory for {deployment_name}"
                self._record_action(f"pod/{namespace}/{pod_name}", "increase_memory", True, details)
                return {'action_taken': True, 'action': 'increase_resources', 'details': details}
            safe_api_call(lambda: self.k8s_api.delete_namespaced_pod(name=pod_name, namespace=namespace))
            details = f"Restarted {pod_name} due to OOM"
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
                current_replicas = deployment.spec.replicas or 1
                new_replicas = min(current_replicas + 1, 10)  # Cap at 10 to avoid over-scaling
                deployment.spec.replicas = new_replicas
                safe_api_call(lambda: self.k8s_apps_api.patch_namespaced_deployment(
                    name=deployment_name, namespace=namespace, body=deployment), max_retries=3)
                details = f"Scaled {deployment_name} from {current_replicas} to {new_replicas} replicas due to resource exhaustion"
                self._record_action(f"pod/{namespace}/{pod_name}", "scale_deployment", True, details)
                return {'action_taken': True, 'action': 'scale_deployment', 'details': details}
            
            # Fallback: Restart pod if no deployment
            safe_api_call(lambda: self.k8s_api.delete_namespaced_pod(name=pod_name, namespace=namespace))
            details = f"Restarted {pod_name} due to resource exhaustion (no deployment found)"
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

    def remediate_issue(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        resource_id = f"{prediction['resource_type']}/{prediction['namespace']}/{prediction['resource_name']}"
        if self._is_in_cooldown(resource_id):
            self.logger.info(f"{resource_id} in cooldown, no action")
            return {'action_taken': False, 'reason': 'In cooldown period'}
        
        if prediction['resource_type'] == 'pod':
            issue_type = prediction.get('issue_type')
            if issue_type == 'oom_kill':
                result = self._remediate_pod_oom(prediction)
            elif issue_type == 'crash_loop':
                result = self._remediate_pod_crash_loop(prediction)
            elif issue_type == 'unknown':
                result = self._remediate_pod_unknown_state(prediction)
            elif issue_type == 'pending':
                result = self.remediate_pending_pod(prediction['namespace'], prediction['resource_name'])
            elif issue_type == 'resource_exhaustion':
                result = self._remediate_resource_exhaustion(prediction)
            elif issue_type == 'network_issue':
                result = self._remediate_network_issue(prediction)
            elif issue_type == 'partial_failure':
                result = self._remediate_partial_failure(prediction)
            elif issue_type == 'io_issue':
                result = self._remediate_io_issue(prediction)
            else:
                self.logger.warning(f"No remediation for {issue_type}")
                return {'action_taken': False, 'reason': 'No remediation defined'}
        else:
            self.logger.warning(f"Unsupported resource type: {prediction['resource_type']}")
            return {'action_taken': False, 'reason': 'Unsupported resource type'}
        
        effectiveness = self._evaluate_remediation_effectiveness(resource_id, prediction)
        self.logger.info(f"Effectiveness evaluation for {resource_id}: {effectiveness}")
        return result

    def monitor_cluster(self, interval=10):  # Reduced interval for faster testing
        self.logger.info("Starting real-time cluster monitoring")
        w = watch.Watch()
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
                        prediction_df = anomaly_prediction.predict_anomalies(metrics_seq, sequence_length)
                        self.logger.debug(f"Prediction for {pod_id}: {prediction_df.to_dict()}")
                        if not prediction_df.empty and prediction_df['predicted_anomaly'].iloc[0] == 1:
                            prediction = {
                                'resource_type': 'pod',
                                'resource_name': pod.metadata.name,
                                'namespace': pod.metadata.namespace,
                                'issue_type': self._map_anomaly_type_to_issue(
                                    prediction_df['anomaly_type'].iloc[0], pod.status.phase, metrics),
                                'confidence': prediction_df['anomaly_probability'].iloc[0],
                                'metrics': metrics
                            }
                            result = self.remediate_issue(prediction)
                            self.logger.info(f"Processed {pod_id}: {result}")
                        else:
                            self.logger.info(f"Processed {pod_id}: {{'action_taken': False, 'details': 'No anomaly detected'}}")
                    else:
                        self.logger.debug(f"Waiting for {sequence_length} samples for {pod_id}, current: {len(self.pod_history[pod_id])}")
                except Exception as e:
                    self.logger.error(f"Error processing event for {pod_id}: {str(e)}")
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