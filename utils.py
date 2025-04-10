import logging
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("k8s-remediation-utils")

def setup_k8s_client():
    from kubernetes import client, config
    try:
        config.load_kube_config()
    except Exception:
        config.load_incluster_config()
    return client.CoreV1Api(), client.AppsV1Api()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_api_call(call):
    return call()

def parse_resource_value(value: str, factor: float = 1.0) -> str:
    if value.endswith('i'):
        unit = value[-2:]
        val = float(value[:-2])
        new_val = val * factor
        return f"{int(new_val)}{unit}"
    elif value.endswith('m'):
        val = float(value[:-1])
        new_val = val * factor
        return f"{int(new_val)}m"
    else:
        try:
            val = float(value)
            new_val = val * factor
            return str(new_val)
        except ValueError:
            logger.warning(f"Unable to parse resource value: {value}, returning original")
            return value