from .base_agent import BaseAgent
from langchain.agents import Tool
from typing import List, Dict, Any
import kubernetes as k8s
import os
from datetime import datetime, timedelta
import re
import json

class LogAgent(BaseAgent):
    """Agent responsible for managing and analyzing logs in the Kubernetes cluster"""
    
    def __init__(self, llm):
        """Initialize the log agent"""
        super().__init__(llm)
        self.k8s_client = k8s.client.CoreV1Api()
    
    def get_tools(self) -> List[Tool]:
        """Get the log management tools available to this agent"""
        return [
            Tool(
                name="get_pod_logs",
                func=self._get_pod_logs,
                description="Get logs from a pod"
            ),
            Tool(
                name="analyze_logs",
                func=self._analyze_logs,
                description="Analyze logs for patterns and issues"
            ),
            Tool(
                name="search_logs",
                func=self._search_logs,
                description="Search logs for specific patterns or terms"
            ),
            Tool(
                name="get_container_logs",
                func=self._get_container_logs,
                description="Get logs from a specific container in a pod"
            )
        ]
    
    def _get_pod_logs(self, pod_name: str, namespace: str = "default", 
                     tail_lines: int = 100, previous: bool = False) -> Dict[str, Any]:
        """Get logs from a pod"""
        try:
            # Get pod logs
            logs = self.k8s_client.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=tail_lines,
                previous=previous
            )
            
            # Split logs into lines
            log_lines = logs.split('\n')
            
            return {
                'pod_name': pod_name,
                'namespace': namespace,
                'log_lines': log_lines,
                'count': len(log_lines)
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _analyze_logs(self, pod_name: str, namespace: str = "default", 
                     tail_lines: int = 1000) -> Dict[str, Any]:
        """Analyze logs for patterns and issues"""
        try:
            # Get pod logs
            logs = self._get_pod_logs(pod_name, namespace, tail_lines)
            
            if 'error' in logs:
                return logs
            
            log_lines = logs['log_lines']
            
            # Common error patterns
            error_patterns = {
                'error': r'(?i)error|exception|failed|failure|timeout|deadlock|panic|crash',
                'warning': r'(?i)warning|warn|deprecated|deprecation',
                'connection': r'(?i)connection refused|connection reset|connection timed out|no route to host',
                'permission': r'(?i)permission denied|access denied|unauthorized|forbidden',
                'resource': r'(?i)out of memory|out of disk space|resource quota exceeded|quota exceeded',
                'crash': r'(?i)segmentation fault|core dumped|killed|oom|out of memory'
            }
            
            # Analyze logs for patterns
            analysis = {
                'pod_name': pod_name,
                'namespace': namespace,
                'total_lines': len(log_lines),
                'patterns': {},
                'error_count': 0,
                'warning_count': 0,
                'last_errors': [],
                'last_warnings': []
            }
            
            # Count occurrences of each pattern
            for pattern_name, pattern in error_patterns.items():
                matches = []
                for i, line in enumerate(log_lines):
                    if re.search(pattern, line):
                        matches.append({
                            'line_number': i + 1,
                            'content': line
                        })
                
                analysis['patterns'][pattern_name] = {
                    'count': len(matches),
                    'matches': matches[-5:] if matches else []  # Keep last 5 matches
                }
                
                if pattern_name == 'error':
                    analysis['error_count'] = len(matches)
                    analysis['last_errors'] = matches[-5:] if matches else []
                elif pattern_name == 'warning':
                    analysis['warning_count'] = len(matches)
                    analysis['last_warnings'] = matches[-5:] if matches else []
            
            # Try to detect application-specific issues
            app_issues = self._detect_application_issues(log_lines)
            analysis['application_issues'] = app_issues
            
            return analysis
        except Exception as e:
            return self.handle_error(e)
    
    def _search_logs(self, pod_name: str, search_term: str, namespace: str = "default", 
                    tail_lines: int = 1000, case_sensitive: bool = False) -> Dict[str, Any]:
        """Search logs for specific patterns or terms"""
        try:
            # Get pod logs
            logs = self._get_pod_logs(pod_name, namespace, tail_lines)
            
            if 'error' in logs:
                return logs
            
            log_lines = logs['log_lines']
            
            # Search for the term
            matches = []
            flags = 0 if case_sensitive else re.IGNORECASE
            
            for i, line in enumerate(log_lines):
                if re.search(search_term, line, flags):
                    matches.append({
                        'line_number': i + 1,
                        'content': line
                    })
            
            return {
                'pod_name': pod_name,
                'namespace': namespace,
                'search_term': search_term,
                'case_sensitive': case_sensitive,
                'matches': matches,
                'count': len(matches)
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _get_container_logs(self, pod_name: str, container_name: str, namespace: str = "default", 
                          tail_lines: int = 100, previous: bool = False) -> Dict[str, Any]:
        """Get logs from a specific container in a pod"""
        try:
            # Get container logs
            logs = self.k8s_client.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                container=container_name,
                tail_lines=tail_lines,
                previous=previous
            )
            
            # Split logs into lines
            log_lines = logs.split('\n')
            
            return {
                'pod_name': pod_name,
                'container_name': container_name,
                'namespace': namespace,
                'log_lines': log_lines,
                'count': len(log_lines)
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _detect_application_issues(self, log_lines: List[str]) -> List[Dict[str, Any]]:
        """Detect application-specific issues from logs"""
        issues = []
        
        # Common application issues patterns
        app_patterns = {
            'database_connection': {
                'pattern': r'(?i)database connection|sql error|database error|connection pool|connection refused',
                'severity': 'high',
                'description': 'Database connection issues detected'
            },
            'api_error': {
                'pattern': r'(?i)api error|http error|status code 5|status code 4|request failed',
                'severity': 'medium',
                'description': 'API errors detected'
            },
            'authentication': {
                'pattern': r'(?i)authentication failed|login failed|invalid credentials|token expired',
                'severity': 'high',
                'description': 'Authentication issues detected'
            },
            'configuration': {
                'pattern': r'(?i)configuration error|config error|missing config|invalid config',
                'severity': 'medium',
                'description': 'Configuration issues detected'
            },
            'dependency': {
                'pattern': r'(?i)dependency error|service unavailable|service not found|circuit breaker',
                'severity': 'medium',
                'description': 'Dependency issues detected'
            }
        }
        
        # Check for each pattern
        for issue_type, issue_info in app_patterns.items():
            matches = []
            for i, line in enumerate(log_lines):
                if re.search(issue_info['pattern'], line):
                    matches.append({
                        'line_number': i + 1,
                        'content': line
                    })
            
            if matches:
                issues.append({
                    'type': issue_type,
                    'severity': issue_info['severity'],
                    'description': issue_info['description'],
                    'matches': matches[-3:] if matches else []  # Keep last 3 matches
                })
        
        return issues
    
    def process_result(self, result: str) -> Dict[str, Any]:
        """Process the result from log management actions"""
        # TODO: Implement result processing logic
        return {
            'status': 'processed',
            'result': result
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for log management"""
        required_fields = ['action', 'target']
        return all(field in input_data for field in required_fields)
    
    def format_output(self, output_data: Dict[str, Any]) -> str:
        """Format the log management output data"""
        # TODO: Implement output formatting logic
        return str(output_data) 