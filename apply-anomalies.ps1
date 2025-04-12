# Function to apply a manifest and wait for it to take effect
function Apply-AndWait {
    param (
        [string]$manifestFile
    )
    
    Write-Host "Applying $manifestFile..."
    kubectl apply -f $manifestFile
    
    Write-Host "Waiting for pod to be ready..."
    kubectl wait --for=condition=available deployment/test-anomaly-deployment --timeout=60s
    
    Write-Host "Waiting for anomaly to be detected..."
    Start-Sleep -Seconds 30
    
    Write-Host "Cleaning up..."
    kubectl delete -f $manifestFile
    Start-Sleep -Seconds 10
}

# Start with the base deployment
Write-Host "Creating base deployment..."
kubectl apply -f test-deployment.yaml
Start-Sleep -Seconds 10

# Apply each anomaly in sequence
Apply-AndWait -manifestFile "anomaly-oom.yaml"
Apply-AndWait -manifestFile "anomaly-crash-loop.yaml"
Apply-AndWait -manifestFile "anomaly-resource-exhaustion.yaml"
Apply-AndWait -manifestFile "anomaly-network.yaml"

# Clean up
Write-Host "Cleaning up base deployment..."
kubectl delete -f test-deployment.yaml 