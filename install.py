#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")

def create_virtual_environment():
    """Create a virtual environment"""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("Virtual environment already exists")
        return
    
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    print("Virtual environment created successfully")

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    
    # Determine the pip path based on the platform
    if platform.system() == "Windows":
        pip_path = Path(".venv/Scripts/pip.exe")
    else:
        pip_path = Path(".venv/bin/pip")
    
    # Upgrade pip
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    print("Requirements installed successfully")

def setup_environment_file():
    """Create or update the .env file"""
    env_path = Path(".env")
    if env_path.exists():
        print(".env file already exists, skipping creation")
        return
    
    print("Creating .env file...")
    with open(env_path, "w") as f:
        f.write("""# LLM API Key for Llama 3.1 Nemotron 70B Instruct
LLAMA_API_KEY=your_llama_api_key_here

# LLM Configuration
LLM_MODEL=nemotron-70b-instruct
LLM_API_BASE=https://api.nvcf.nvidia.com/v1

# Kubernetes configuration
KUBERNETES_CONFIG_PATH=~/.kube/config

# Prometheus configuration
PROMETHEUS_URL=http://localhost:9090

# Logging configuration
LOG_LEVEL=INFO

# Agent configuration
AGENT_MEMORY_SIZE=1000
AGENT_TEMPERATURE=0.7

# Monitoring configuration
MONITORING_INTERVAL=60  # seconds
METRICS_HISTORY_SIZE=10
REMEDIATION_COOLDOWN=300  # seconds

# Resource thresholds
CPU_THRESHOLD=80  # percentage
MEMORY_THRESHOLD=80  # percentage
NETWORK_DROP_THRESHOLD=10  # packets per second
""")
    print(".env file created successfully")
    print("Please update the .env file with your actual configuration values")

def main():
    """Main installation function"""
    print("Starting installation of Kubernetes Multi-Agent System...")
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install requirements
    install_requirements()
    
    # Setup environment file
    setup_environment_file()
    
    print("\nInstallation completed successfully!")
    print("\nNext steps:")
    print("1. Update the .env file with your actual configuration values")
    print("2. Run the system using: python -m mas")
    print("\nFor more information, please refer to the documentation")

if __name__ == "__main__":
    main() 