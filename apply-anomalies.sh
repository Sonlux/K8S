#!/bin/bash

# Function to apply a manifest and wait for it to take effect
apply_and_wait() {
    echo "Applying $1..."
    kubectl apply -f $1
    echo "Waiting for pod to be ready..."
    kubectl wait --for=condition=available deployment/test-anomaly-deployment --timeout=60s
    echo "Waiting for anomaly to be detected..."
    sleep 30
    echo "Cleaning up..."
    kubectl delete -f $1
    sleep 10
}

# Start with the base deployment
echo "Creating base deployment..."
kubectl apply -f test-deployment.yaml
sleep 10

# Apply each anomaly in sequence
apply_and_wait anomaly-oom.yaml
apply_and_wait anomaly-crash-loop.yaml
apply_and_wait anomaly-resource-exhaustion.yaml
apply_and_wait anomaly-network.yaml

# Clean up
echo "Cleaning up base deployment..."
kubectl delete -f test-deployment.yaml 