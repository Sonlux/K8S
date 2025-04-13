# Kubernetes Multi-Agent System (MAS)

An intelligent Multi-Agent System for monitoring, analyzing, and remediating issues in Kubernetes clusters.

## Key Features

- **Intelligent Monitoring**: Continuous monitoring of Kubernetes pod metrics
- **Issue Detection**: Automatic detection of resource exhaustion, crash loops, and network issues
- **Predictive Scaling**: ML-based prediction of future resource needs
- **Smart Remediation**: Automated remediation actions with rollback capability
- **Effectiveness Evaluation**: Tracking and reporting of remediation effectiveness

## System Components

- **Coordinator**: Orchestrates monitoring and remediation actions
- **Specialized Agents**: Handle specific issue types
- **Remediation System**: Implements remediation strategies with rollback
- **Effectiveness Evaluation**: Measures the success of remediation actions
- **Metrics Collection**: Gathers pod and cluster metrics

## Getting Started

### Prerequisites

- Kubernetes Cluster with metrics-server installed
- Python 3.8 or higher
- Access to Kubernetes API (kubeconfig)

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/k8s-mas.git
cd k8s-mas
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
# Create a .env file
touch .env

# Add your configuration
echo "KUBERNETES_CONTEXT=your-k8s-context" >> .env
echo "MONITOR_INTERVAL=300" >> .env  # 5 minutes in seconds
```

### Running the System

To start the Multi-Agent System:

```bash
python -m mas
```

This will start the monitoring loop and begin detecting and remediating issues.

## Architecture

The MAS consists of several components:

1. **Coordinator (`coordinator.py`)**: The central component that monitors the cluster and coordinates remediation actions.
2. **Specialized Agents (`specialized_agents.py`)**: Specialized agents for handling different types of issues:
   - ResourceExhaustionAgent: Handles CPU and memory issues
   - NetworkIssueAgent: Handles network connectivity issues
   - CrashLoopAgent: Handles crash loops and restarts
   - AnalysisAgent: Provides deeper analysis of issues
3. **Remediation System (`remediation.py`)**: Implements strategies for remediating different types of issues.
4. **Effectiveness Evaluation (`remediation_effectiveness.py`)**: Tracks the effectiveness of remediation actions.

## Phase 2 Enhancements

### Remediation Actions

- **Predictive Scaling**: Uses historical metrics to predict future resource needs
- **Adaptive HPA Configuration**: Dynamically adjusts HPA settings based on workload patterns
- **Resource Optimization**: Intelligently adjusts CPU and memory limits based on actual usage
- **Rollback Capability**: Automatically rolls back failed remediation actions

### Effectiveness Evaluation

- **Success Tracking**: Measures and reports on remediation success rates
- **Trend Analysis**: Analyzes effectiveness trends over time
- **Strategy Optimization**: Identifies the most effective remediation strategies

## Configuration

The system can be configured through environment variables:

| Variable                 | Description                                           | Default         |
| ------------------------ | ----------------------------------------------------- | --------------- |
| KUBERNETES_CONTEXT       | Kubernetes context to use                             | current-context |
| MONITOR_INTERVAL         | Interval between monitoring cycles (seconds)          | 300             |
| COOLDOWN_PERIOD          | Cooldown period between remediation actions (seconds) | 300             |
| MAX_REMEDIATION_ATTEMPTS | Maximum remediation attempts                          | 3               |
| LOG_LEVEL                | Logging level (INFO, DEBUG, WARNING, ERROR)           | INFO            |

## Extending the System

The MAS is designed to be extensible. You can add:

1. New specialized agents by extending the base agent class
2. New remediation strategies by adding methods to the RemediationSystem
3. Custom metrics by modifying the metrics collection logic

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kubernetes community for the excellent API and documentation
- LangChain and LangGraph for agent orchestration capabilities
