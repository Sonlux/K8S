import logging
import time
from datetime import datetime
from remediation_logic import K8sRemediation, generate_remediation_recommendations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('remediation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('k8s-remediation-runner')

def main():
    try:
        # Initialize the remediation system
        logger.info("Initializing Kubernetes Remediation System...")
        remediation = K8sRemediation(
            cooldown_period=300,  # 5 minutes between actions
            max_cooldown_period=3600,  # 1 hour max cooldown
            confidence_threshold=0.8  # 80% confidence required
        )
        
        # Start monitoring the cluster
        logger.info("Starting cluster monitoring...")
        remediation.monitor_cluster(interval=10)  # Check every 10 seconds
        
        # Keep the script running
        while True:
            try:
                # Get current cluster state
                cluster_state = remediation.get_cluster_state()
                
                # Process each pod that needs attention
                for pod in cluster_state.get('pods', []):
                    if pod.get('needs_attention'):
                        # Create prediction object for the pod
                        prediction = {
                            'resource_type': 'pod',
                            'namespace': pod['namespace'],
                            'resource_name': pod['name'],
                            'metrics': {
                                'CPU Usage (%)': pod.get('cpu_usage', 0.0),
                                'Memory Usage (%)': pod.get('memory_usage', 0.0)
                            },
                            'predicted_metrics': pod.get('predicted_metrics', {}),
                            'issue_type': pod.get('issue_type', 'unknown')
                        }
                        
                        # Get remediation recommendations
                        recommendations = generate_remediation_recommendations(
                            anomaly_results=pod.get('anomaly_results', {}),
                            historical_usage=pod.get('historical_usage', []),
                            current_resources=pod.get('current_resources', {}),
                            namespace=pod['namespace'],
                            pod_name=pod['name']
                        )
                        
                        # Log recommendations
                        logger.info(f"Recommendations for {pod['namespace']}/{pod['name']}:")
                        logger.info(f"Anomaly Score: {recommendations['anomaly_score']}")
                        logger.info(f"Action Required: {recommendations['recommendations'].get('cpu', {}).get('action_required')}")
                        
                        # Take remediation action if needed
                        if recommendations['is_anomaly']:
                            logger.info(f"Taking remediation action for {pod['namespace']}/{pod['name']}")
                            result = remediation.remediate_issue(prediction)
                            logger.info(f"Remediation result: {result}")
                
                # Sleep for a short interval
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error during monitoring cycle: {str(e)}")
                time.sleep(30)  # Wait longer on error
                
    except KeyboardInterrupt:
        logger.info("Shutting down remediation system...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 