import requests
from typing import Dict, Any, Optional
import logging

class PrometheusClient:
    """Wrapper for Prometheus HTTP API"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        """Initialize the Prometheus client"""
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger("mas-prometheus")
    
    def query(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute a PromQL query"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/query",
                params={'query': query}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error querying Prometheus: {str(e)}")
            return None
    
    def query_range(self, query: str, start: int, end: int, step: str = "1m") -> Optional[Dict[str, Any]]:
        """Execute a range query"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/query_range",
                params={
                    'query': query,
                    'start': start,
                    'end': end,
                    'step': step
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error querying Prometheus range: {str(e)}")
            return None 