#!/usr/bin/env python3

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from kubernetes import client
import numpy as np
import pandas as pd
from collections import defaultdict

class RemediationEffectiveness:
    """Track and evaluate the effectiveness of remediation actions."""
    
    def __init__(self, k8s_api: client.CoreV1Api, apps_api: client.AppsV1Api,
                 metrics_api: client.CustomObjectsApi):
        """Initialize the remediation effectiveness tracker."""
        self.logger = logging.getLogger("mas-effectiveness")
        self.k8s_api = k8s_api
        self.apps_api = apps_api
        self.metrics_api = metrics_api
        
        # Tracking structures
        self.pre_remediation_metrics = {}  # Metrics before remediation
        self.post_remediation_metrics = {}  # Metrics after remediation
        self.remediation_outcomes = []  # List of all remediation outcomes
        self.effectiveness_by_strategy = defaultdict(list)  # Effectiveness scores by strategy
        
        # Configuration
        self.evaluation_window = timedelta(minutes=10)  # How long to monitor after remediation
        self.significance_threshold = 10.0  # Minimum percentage improvement to be significant
        
    def record_pre_remediation_state(self, pod_id: str, issue_type: str, metrics: Dict[str, Any]) -> None:
        """Record the state of a pod before remediation."""
        self.logger.info(f"Recording pre-remediation state for {pod_id} ({issue_type})")
        
        # Store metrics with timestamp
        self.pre_remediation_metrics[pod_id] = {
            'timestamp': datetime.now(),
            'issue_type': issue_type,
            'metrics': metrics,
            'evaluated': False
        }
    
    def record_remediation_action(self, pod_id: str, action: str, 
                                  parameters: Dict[str, Any]) -> None:
        """Record a remediation action that was taken."""
        if pod_id not in self.pre_remediation_metrics:
            self.logger.warning(f"No pre-remediation state recorded for {pod_id}")
            return
            
        self.pre_remediation_metrics[pod_id]['action'] = action
        self.pre_remediation_metrics[pod_id]['parameters'] = parameters
        self.pre_remediation_metrics[pod_id]['action_timestamp'] = datetime.now()
    
    def record_post_remediation_state(self, pod_id: str, metrics: Dict[str, Any]) -> None:
        """Record the state of a pod after remediation."""
        if pod_id not in self.pre_remediation_metrics:
            self.logger.warning(f"No pre-remediation state recorded for {pod_id}")
            return
            
        # Don't evaluate if already evaluated or if no action was taken
        if self.pre_remediation_metrics[pod_id].get('evaluated', False) or \
           'action' not in self.pre_remediation_metrics[pod_id]:
            return
            
        # Store post-remediation metrics
        self.post_remediation_metrics[pod_id] = {
            'timestamp': datetime.now(),
            'metrics': metrics
        }
        
        # Calculate effectiveness
        effectiveness = self._calculate_effectiveness(pod_id)
        
        # Update tracking structures
        action = self.pre_remediation_metrics[pod_id].get('action', 'unknown')
        issue_type = self.pre_remediation_metrics[pod_id].get('issue_type', 'unknown')
        
        # Record to outcomes list
        outcome = {
            'pod_id': pod_id,
            'issue_type': issue_type,
            'action': action,
            'effectiveness': effectiveness,
            'pre_remediation': self.pre_remediation_metrics[pod_id],
            'post_remediation': self.post_remediation_metrics[pod_id],
            'timestamp': datetime.now()
        }
        self.remediation_outcomes.append(outcome)
        
        # Update effectiveness by strategy
        strategy_key = f"{issue_type}:{action}"
        self.effectiveness_by_strategy[strategy_key].append(effectiveness)
        
        # Mark as evaluated
        self.pre_remediation_metrics[pod_id]['evaluated'] = True
        
        self.logger.info(f"Remediation effectiveness for {pod_id}: {effectiveness:.2f}%")
    
    def _calculate_effectiveness(self, pod_id: str) -> float:
        """Calculate the effectiveness of a remediation action."""
        pre = self.pre_remediation_metrics[pod_id]
        post = self.post_remediation_metrics[pod_id]
        
        # Determine which metrics to compare based on issue type
        issue_type = pre.get('issue_type', 'unknown')
        action = pre.get('action', 'unknown')
        
        # Calculate improvement
        if issue_type == 'resource_exhaustion':
            # For resource issues, compare CPU and memory usage
            pre_cpu = pre['metrics'].get('cpu_usage', 0)
            post_cpu = post['metrics'].get('cpu_usage', 0)
            pre_mem = pre['metrics'].get('memory_usage', 0)
            post_mem = post['metrics'].get('memory_usage', 0)
            
            # Calculate percentage improvements (negative is better)
            cpu_improvement = ((pre_cpu - post_cpu) / pre_cpu * 100) if pre_cpu > 0 else 0
            mem_improvement = ((pre_mem - post_mem) / pre_mem * 100) if pre_mem > 0 else 0
            
            # Weight CPU and memory equally
            return (cpu_improvement + mem_improvement) / 2
            
        elif issue_type == 'crash_loop':
            # For crash loops, check if the pod is stable
            pre_restarts = pre['metrics'].get('restart_count', 0)
            post_restarts = post['metrics'].get('restart_count', 0)
            
            # If restarts haven't increased, that's good
            if post_restarts <= pre_restarts:
                return 100.0  # 100% effective
            else:
                return -100.0  # Negative effectiveness - got worse
                
        elif issue_type == 'network_issue':
            # For network issues, check network errors
            pre_errors = pre['metrics'].get('network_errors', 0)
            post_errors = post['metrics'].get('network_errors', 0)
            
            # Calculate percentage reduction in errors
            if pre_errors > 0:
                return ((pre_errors - post_errors) / pre_errors * 100)
            elif post_errors == 0:
                return 100.0  # No errors before or after
            else:
                return -100.0  # Had no errors before, but have some now
        
        else:
            # For unknown issue types, use a generic approach
            # Check if the pod is Running and hasn't restarted
            pre_phase = pre['metrics'].get('phase', 'Unknown')
            post_phase = post['metrics'].get('phase', 'Unknown')
            
            if pre_phase != 'Running' and post_phase == 'Running':
                return 100.0  # Pod is now running
            elif pre_phase == 'Running' and post_phase == 'Running':
                return 50.0  # No change but still running
            else:
                return 0.0  # No improvement
    
    def get_effectiveness_summary(self) -> Dict[str, Any]:
        """Get a summary of remediation effectiveness."""
        summary = {
            'total_remediations': len(self.remediation_outcomes),
            'average_effectiveness': 0.0,
            'by_issue_type': {},
            'by_action': {},
            'by_strategy': {}
        }
        
        if not self.remediation_outcomes:
            return summary
            
        # Calculate overall average
        effectiveness_scores = [o['effectiveness'] for o in self.remediation_outcomes]
        summary['average_effectiveness'] = np.mean(effectiveness_scores)
        
        # Group by issue type
        issue_types = {}
        for outcome in self.remediation_outcomes:
            issue_type = outcome['issue_type']
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(outcome['effectiveness'])
            
        for issue_type, scores in issue_types.items():
            summary['by_issue_type'][issue_type] = {
                'count': len(scores),
                'average_effectiveness': np.mean(scores),
                'min_effectiveness': min(scores),
                'max_effectiveness': max(scores)
            }
            
        # Group by action
        actions = {}
        for outcome in self.remediation_outcomes:
            action = outcome['action']
            if action not in actions:
                actions[action] = []
            actions[action].append(outcome['effectiveness'])
            
        for action, scores in actions.items():
            summary['by_action'][action] = {
                'count': len(scores),
                'average_effectiveness': np.mean(scores),
                'min_effectiveness': min(scores),
                'max_effectiveness': max(scores)
            }
            
        # Group by strategy (issue_type + action)
        for strategy, scores in self.effectiveness_by_strategy.items():
            summary['by_strategy'][strategy] = {
                'count': len(scores),
                'average_effectiveness': np.mean(scores),
                'min_effectiveness': min(scores),
                'max_effectiveness': max(scores)
            }
            
        return summary
    
    def get_effectiveness_trend(self, window_days: int = 7) -> Dict[str, Any]:
        """Get the trend of remediation effectiveness over time."""
        if not self.remediation_outcomes:
            return {'trend': 'No data'}
            
        # Convert to DataFrame for easier analysis
        data = []
        for outcome in self.remediation_outcomes:
            data.append({
                'timestamp': outcome['timestamp'],
                'effectiveness': outcome['effectiveness'],
                'issue_type': outcome['issue_type'],
                'action': outcome['action']
            })
            
        df = pd.DataFrame(data)
        
        # Filter to window
        start_date = datetime.now() - timedelta(days=window_days)
        df = df[df['timestamp'] >= start_date]
        
        if len(df) < 2:
            return {'trend': 'Insufficient data'}
            
        # Resample by day
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date')['effectiveness'].mean().reset_index()
        
        # Calculate trend
        x = np.arange(len(daily))
        y = daily['effectiveness'].values
        
        # Simple linear regression
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            trend_direction = 'improving' if slope > 0 else 'worsening' if slope < 0 else 'stable'
            trend_magnitude = abs(slope)
            
            return {
                'trend': trend_direction,
                'magnitude': trend_magnitude,
                'daily_values': daily.to_dict(orient='records'),
                'slope': slope,
                'intercept': intercept
            }
        else:
            return {'trend': 'Insufficient data'}
    
    def get_best_strategy(self, issue_type: str) -> Optional[Dict[str, Any]]:
        """Get the most effective strategy for a given issue type."""
        # Find all strategies that apply to this issue type
        applicable_strategies = {}
        
        for strategy, scores in self.effectiveness_by_strategy.items():
            strategy_issue_type, action = strategy.split(':', 1)
            if strategy_issue_type == issue_type and len(scores) >= 3:  # Require at least 3 data points
                applicable_strategies[action] = {
                    'average_effectiveness': np.mean(scores),
                    'count': len(scores),
                    'confidence': 1.0 - (1.0 / len(scores))  # More samples = higher confidence
                }
        
        if not applicable_strategies:
            return None
            
        # Sort by effectiveness and confidence
        sorted_strategies = sorted(
            applicable_strategies.items(),
            key=lambda x: (x[1]['average_effectiveness'] * x[1]['confidence']),
            reverse=True
        )
        
        best_action, stats = sorted_strategies[0]
        
        return {
            'issue_type': issue_type,
            'best_action': best_action,
            'effectiveness': stats['average_effectiveness'],
            'confidence': stats['confidence'],
            'sample_size': stats['count']
        }
    
    def schedule_evaluation_task(self, pod_id: str) -> None:
        """Schedule an evaluation task for a pod that was remediated."""
        # This method would typically add the pod_id to a queue to be processed
        # by a background task that collects post-remediation metrics after a delay
        pass
    
    def cleanup_old_records(self, max_age_days: int = 30) -> None:
        """Clean up old remediation records."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Clean up outcomes
        self.remediation_outcomes = [
            o for o in self.remediation_outcomes 
            if o['timestamp'] >= cutoff_date
        ]
        
        # Clean up pre-remediation metrics
        for pod_id in list(self.pre_remediation_metrics.keys()):
            if self.pre_remediation_metrics[pod_id]['timestamp'] < cutoff_date:
                del self.pre_remediation_metrics[pod_id]
                
        # Clean up post-remediation metrics
        for pod_id in list(self.post_remediation_metrics.keys()):
            if self.post_remediation_metrics[pod_id]['timestamp'] < cutoff_date:
                del self.post_remediation_metrics[pod_id] 