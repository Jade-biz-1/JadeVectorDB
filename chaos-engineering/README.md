# Chaos Engineering Framework for JadeVectorDB

## Overview

The Chaos Engineering Framework for JadeVectorDB is designed to test system resilience by intentionally introducing failures and observing how the system responds. This framework enables systematic testing of the distributed vector database under various failure conditions.

Chaos engineering follows the principle of "breaking things on purpose" to build confidence in the system's ability to withstand turbulent conditions in production.

## Components

The chaos engineering framework consists of the following components:

1. **Network Partition Simulation**: Simulates network failures between nodes
2. **Node Failure Injection**: Simulates various types of node failures
3. **Resource Exhaustion Simulation**: Simulates CPU, memory, disk, and network exhaustion
4. **Automated Chaos Experiment Execution**: Orchestrates complex chaos experiments
5. **Monitoring Integration**: Tracks system metrics during chaos experiments
6. **Test Scripts**: Provides comprehensive testing capabilities

## Architecture

The framework is organized into the following directories:

```
chaos-engineering/
├── network/                 # Network partition simulation
├── node/                    # Node failure injection
├── resource/                # Resource exhaustion simulation
├── automation/              # Automated experiment execution
├── monitoring/              # Monitoring integration
├── scripts/                 # Test and utility scripts
├── experiments/             # Experiment definition files
└── README.md               # This documentation
```

## Prerequisites

Before using the chaos engineering framework, ensure you have:

- Kubernetes cluster with JadeVectorDB deployed
- `kubectl` installed and configured
- `jq` installed (for JSON processing)
- Adequate cluster permissions for pod manipulation
- Monitoring stack (Prometheus) deployed for metrics tracking

## Usage

### 1. Network Partition Simulation

Simulates network partitions between nodes in the cluster:

```bash
# Random network partition (20% of nodes)
./network/simulate-network-partition.sh [namespace] random [duration] [pod-list]

# Partition specific pods
./network/simulate-network-partition.sh jadevectordb-system specific 60 pod1,pod2,pod3

# Partition all nodes
./network/simulate-network-partition.sh jadevectordb-system all 120
```

### 2. Node Failure Injection

Simulates various node failure scenarios:

```bash
# Delete random pods
./node/simulate-node-failure.sh [namespace] pod-delete [duration] [pod-list]

# Terminate random pods
./node/simulate-node-failure.sh jadevectordb-system pod-kill 120

# Make pods crashloop
./node/simulate-node-failure.sh jadevectordb-system crashloop 180

# Drain a node (requires admin privileges)
./node/simulate-node-failure.sh jadevectordb-system node-drain 120 nodename
```

### 3. Resource Exhaustion Simulation

Simulates resource exhaustion on nodes:

```bash
# Memory exhaustion (medium intensity)
./resource/simulate-resource-exhaustion.sh [namespace] memory [duration] medium [pod-list]

# CPU exhaustion (high intensity)
./resource/simulate-resource-exhaustion.sh jadevectordb-system cpu 120 high

# Disk exhaustion (low intensity)
./resource/simulate-resource-exhaustion.sh jadevectordb-system disk 60 low

# Network exhaustion (medium intensity)
./resource/simulate-resource-exhaustion.sh jadevectordb-system network 120 medium
```

### 4. Automated Chaos Experiments

Execute predefined chaos experiment suites:

```bash
# Run default experiment
./automation/execute-chaos-experiment.sh [namespace] [experiment-file] [duration] [grace-period]

# Run stress test experiment
./automation/execute-chaos-experiment.sh jadevectordb-system ./experiments/stress-test-suite.json 600 30
```

### 5. Running Chaos Tests

Execute comprehensive test suites:

```bash
# Run all chaos tests
./scripts/run-chaos-tests.sh [namespace] all [duration] [report-file]

# Run specific test suites
./scripts/run-chaos-tests.sh jadevectordb-system network 300 /tmp/network-test-report.txt
./scripts/run-chaos-tests.sh jadevectordb-system node 300 /tmp/node-test-report.txt
./scripts/run-chaos-tests.sh jadevectordb-system resource 300 /tmp/resource-test-report.txt
./scripts/run-chaos-tests.sh jadevectordb-system automated 300 /tmp/automated-test-report.txt
```

## Experiment Definitions

Experiments are defined in JSON files and specify a sequence of chaos events:

```json
{
  "name": "Example Chaos Experiment Suite",
  "description": "An example chaos experiment suite",
  "experiments": [
    {
      "name": "Memory Exhaustion Test",
      "type": "resource_exhaustion",
      "params": {
        "resource_type": "memory",
        "intensity": "medium"
      },
      "duration": 60,
      "interval": 30
    },
    {
      "name": "Network Partition Test",
      "type": "network_partition",
      "params": {
        "partition_type": "random"
      },
      "duration": 45,
      "interval": 30
    }
  ]
}
```

Available experiment types:
- `resource_exhaustion`: CPU, memory, disk, or network exhaustion
- `network_partition`: Network partition simulation
- `node_failure`: Pod deletion, termination, crashloop, or node drain

## Monitoring Integration

The framework integrates with Prometheus monitoring to track system behavior during chaos experiments:

- **Chaos Experiment Metrics**: Tracks active experiments and their types
- **System Health Metrics**: Monitors error rates, latency, and availability during experiments
- **Alerting Rules**: Predefined alerting rules for detecting issues during chaos

To integrate with your monitoring system:

1. Apply the monitoring configuration:
```bash
kubectl apply -f monitoring/chaos-monitoring.yaml
```

2. The chaos metrics server can expose additional metrics:
```bash
python3 monitoring/chaos-metrics.py --port 8000
```

## Safety Guidelines

To ensure safe use of the chaos engineering framework:

1. **Start Small**: Begin with low-intensity, short-duration experiments
2. **Test in Staging**: Validate experiments in staging environments before production
3. **Monitor Closely**: Always monitor system metrics during experiments
4. **Plan Recovery**: Have a plan to recover quickly if issues arise
5. **Schedule Appropriately**: Run experiments during low-traffic periods
6. **Progress Gradually**: Increase intensity and scope only after successful runs

## Integration with Existing Tests

The chaos engineering framework integrates with existing test infrastructure:

- Compatible with existing integration tests (T188)
- Builds upon security testing infrastructure (T195)
- Can be incorporated into CI/CD pipelines
- Supports automated test execution

## Best Practices

1. **Define Hypothesis**: State what you expect to happen before running experiments
2. **Measure Outcomes**: Capture metrics to validate or disprove your hypothesis
3. **Start with Steady State**: Establish normal system behavior before chaos injection
4. **Vary the Scenarios**: Test different types of failures and conditions
5. **Analyze Results**: Document findings and use them to improve system resilience
6. **Iterate**: Run experiments regularly to test ongoing resilience

## Troubleshooting

### Common Issues

1. **Permissions Error**: Ensure kubectl has necessary permissions for pod manipulation
2. **Resource Limits**: Verify cluster has sufficient resources for chaos experiments
3. **Monitoring Not Working**: Check that Prometheus is properly configured and scraping metrics

### Useful Commands

Monitor ongoing chaos experiments:
```bash
kubectl get pods -n jadevectordb-system
kubectl logs -l app=jadevectordb -n jadevectordb-system -f
```

Check chaos experiment metrics:
```bash
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
# Then visit http://localhost:9090 and search for chaos_experiment_ metrics
```

## Example Workflows

### Basic Resilience Test

1. Run a basic network partition test:
```bash
./scripts/run-chaos-tests.sh jadevectordb-system network 120
```

2. Monitor system behavior during the test:
```bash
kubectl get pods -n jadevectordb-system -w
```

3. Check metrics in Prometheus to validate system behavior

### Comprehensive System Test

1. Create a custom experiment file with multiple failure types
2. Run the automated experiment execution:
```bash
./automation/execute-chaos-experiment.sh jadevectordb-system ./experiments/my-experiment.json 600 30
```

3. Analyze the results and system response

## Security Considerations

- Chaos experiments can affect system availability
- Ensure proper access controls are in place
- Run experiments in isolated environments when possible
- Never run destructive experiments without proper approvals