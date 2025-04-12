import kubernetes as k8s
import time
import random
import logging
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('anomaly-tester')

def setup_k8s_client():
    """Setup Kubernetes client"""
    try:
        config.load_kube_config()
        return client.CoreV1Api(), client.AppsV1Api()
    except Exception as e:
        logger.error(f"Failed to setup Kubernetes client: {str(e)}")
        raise

def create_test_deployment(api, apps_api, namespace="default"):
    """Create a test deployment that we'll use to simulate anomalies"""
    deployment = client.V1Deployment(
        metadata=client.V1ObjectMeta(name="test-anomaly-pod"),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(
                match_labels={"app": "test-anomaly"}
            ),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": "test-anomaly"}
                ),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name="test-container",
                            image="busybox",
                            command=["sh", "-c", "while true; do sleep 1; done"],
                            resources=client.V1ResourceRequirements(
                                requests={
                                    "cpu": "100m",
                                    "memory": "128Mi"
                                },
                                limits={
                                    "cpu": "200m",
                                    "memory": "256Mi"
                                }
                            )
                        )
                    ]
                )
            )
        )
    )
    
    try:
        apps_api.create_namespaced_deployment(
            namespace=namespace,
            body=deployment
        )
        logger.info("Created test deployment")
        time.sleep(5)  # Wait for deployment to be ready
    except ApiException as e:
        logger.error(f"Failed to create deployment: {str(e)}")
        raise

def simulate_oom(namespace="default"):
    """Simulate Out of Memory condition"""
    api, _ = setup_k8s_client()
    try:
        # Get the pod
        pod = api.list_namespaced_pod(
            namespace=namespace,
            label_selector="app=test-anomaly"
        ).items[0]
        
        # Create a container that will consume memory until OOM
        patch = {
            "spec": {
                "containers": [{
                    "name": "test-container",
                    "image": "busybox",
                    "command": ["sh", "-c", "dd if=/dev/zero of=/dev/null bs=1M count=1000"],
                    "resources": {
                        "limits": {
                            "memory": "64Mi"  # Very low memory limit
                        }
                    }
                }]
            }
        }
        
        api.patch_namespaced_pod(
            name=pod.metadata.name,
            namespace=namespace,
            body=patch
        )
        logger.info("Simulated OOM condition")
    except Exception as e:
        logger.error(f"Failed to simulate OOM: {str(e)}")

def simulate_crash_loop(namespace="default"):
    """Simulate Crash Loop condition"""
    api, _ = setup_k8s_client()
    try:
        # Get the pod
        pod = api.list_namespaced_pod(
            namespace=namespace,
            label_selector="app=test-anomaly"
        ).items[0]
        
        # Create a container that will crash immediately
        patch = {
            "spec": {
                "containers": [{
                    "name": "test-container",
                    "image": "busybox",
                    "command": ["sh", "-c", "exit 1"],
                    "resources": {
                        "limits": {
                            "memory": "128Mi"
                        }
                    }
                }]
            }
        }
        
        api.patch_namespaced_pod(
            name=pod.metadata.name,
            namespace=namespace,
            body=patch
        )
        logger.info("Simulated Crash Loop condition")
    except Exception as e:
        logger.error(f"Failed to simulate Crash Loop: {str(e)}")

def simulate_resource_exhaustion(namespace="default"):
    """Simulate Resource Exhaustion"""
    api, _ = setup_k8s_client()
    try:
        # Get the pod
        pod = api.list_namespaced_pod(
            namespace=namespace,
            label_selector="app=test-anomaly"
        ).items[0]
        
        # Create a container that will consume CPU
        patch = {
            "spec": {
                "containers": [{
                    "name": "test-container",
                    "image": "busybox",
                    "command": ["sh", "-c", "while true; do dd if=/dev/zero of=/dev/null bs=1M count=100; done"],
                    "resources": {
                        "limits": {
                            "cpu": "100m",
                            "memory": "128Mi"
                        }
                    }
                }]
            }
        }
        
        api.patch_namespaced_pod(
            name=pod.metadata.name,
            namespace=namespace,
            body=patch
        )
        logger.info("Simulated Resource Exhaustion")
    except Exception as e:
        logger.error(f"Failed to simulate Resource Exhaustion: {str(e)}")

def simulate_network_issue(namespace="default"):
    """Simulate Network Issue"""
    api, _ = setup_k8s_client()
    try:
        # Get the pod
        pod = api.list_namespaced_pod(
            namespace=namespace,
            label_selector="app=test-anomaly"
        ).items[0]
        
        # Create a container that will generate network errors
        patch = {
            "spec": {
                "containers": [{
                    "name": "test-container",
                    "image": "busybox",
                    "command": ["sh", "-c", "while true; do nc -v non-existent-host 80; sleep 1; done"],
                    "resources": {
                        "limits": {
                            "memory": "128Mi"
                        }
                    }
                }]
            }
        }
        
        api.patch_namespaced_pod(
            name=pod.metadata.name,
            namespace=namespace,
            body=patch
        )
        logger.info("Simulated Network Issue")
    except Exception as e:
        logger.error(f"Failed to simulate Network Issue: {str(e)}")

def main():
    namespace = "default"
    api, apps_api = setup_k8s_client()
    
    try:
        # Create test deployment
        create_test_deployment(api, apps_api, namespace)
        
        # Simulate different anomalies with delays between them
        anomalies = [
            ("OOM", simulate_oom),
            ("Crash Loop", simulate_crash_loop),
            ("Resource Exhaustion", simulate_resource_exhaustion),
            ("Network Issue", simulate_network_issue)
        ]
        
        for name, func in anomalies:
            logger.info(f"\nSimulating {name}...")
            func(namespace)
            time.sleep(30)  # Wait for remediation system to detect and fix
            
    except KeyboardInterrupt:
        logger.info("Stopping anomaly simulation...")
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}")
    finally:
        # Cleanup
        try:
            apps_api.delete_namespaced_deployment(
                name="test-anomaly-pod",
                namespace=namespace
            )
            logger.info("Cleaned up test deployment")
        except Exception as e:
            logger.error(f"Failed to cleanup: {str(e)}")

if __name__ == "__main__":
    main() 