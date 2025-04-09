from kubernetes import client, config
import time
import json

def collect_metrics_from_k8s():
    """Collect metrics from Kubernetes API"""
    config.load_incluster_config()  # Load in-cluster configuration
    v1 = client.CoreV1Api()
    
    # Get pods
    pods = v1.list_pod_for_all_namespaces(watch=False)
    
    metrics = []
    for pod in pods.items:
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace
        status = pod.status.phase
        
        # Get metrics for this pod
        # This would typically come from metrics-server or Prometheus
        # For this example, we'll use dummy data
        pod_metrics = {
            'Pod Name': pod_name,
            'Namespace': namespace,
            'Pod Status': status,
            'CPU Usage (%)': 0,  # Would be populated with real data
            'Memory Usage (%)': 0,  # Would be populated with real data
            # Add other metrics
        }
        metrics.append(pod_metrics)
    
    return pd.DataFrame(metrics)
