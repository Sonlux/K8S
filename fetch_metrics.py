import requests
import json
import time
import pandas as pd

# Prometheus API URL
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

# Prometheus queries
QUERY_CPU = {"query": "sum(rate(container_cpu_usage_seconds_total[5m])) by (pod)"}
QUERY_MEMORY = {"query": "sum(container_memory_usage_bytes) by (pod)"}

# CSV file to store collected metrics
CSV_FILE = "prometheus_data.csv"

def fetch_metrics():
    """Fetch CPU and Memory usage from Prometheus and save to CSV"""
    try:
        # Fetch CPU data
        cpu_response = requests.get(PROMETHEUS_URL, params=QUERY_CPU).json()
        memory_response = requests.get(PROMETHEUS_URL, params=QUERY_MEMORY).json()

        # Extract relevant data
        metrics = []
        for i in range(len(cpu_response["data"]["result"])):
            pod_name = cpu_response["data"]["result"][i]["metric"]["pod"]
            cpu_usage = float(cpu_response["data"]["result"][i]["value"][1])
            memory_usage = float(memory_response["data"]["result"][i]["value"][1]) / (1024 * 1024)  # Convert bytes to Mi

            metrics.append({"pod": pod_name, "CPU(m)": cpu_usage, "MEMORY(Mi)": memory_usage})

        # Convert to DataFrame
        df = pd.DataFrame(metrics)

        # Save or append to CSV
        df.to_csv(CSV_FILE, mode='a', index=False, header=not pd.io.common.file_exists(CSV_FILE))
        print(f"✅ Metrics saved to {CSV_FILE}")

    except Exception as e:
        print(f"❌ Error fetching metrics: {e}")

if __name__ == "__main__":
    while True:
        fetch_metrics()
        print("📊 Fetching data... Waiting 60 seconds before next collection.")
        time.sleep(60)  # Fetch data every 60 seconds
