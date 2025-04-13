#!/usr/bin/env python3

import os
import sys
import requests
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_file(url, destination):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress bar
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        logger.info(f"Downloaded {url} to {destination}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def main():
    """Download a small Llama model for testing."""
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Small Llama model URL (about 4GB)
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    model_path = "models/llama-2-7b-chat.gguf"
    
    # Check if model already exists
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        return
    
    # Download model
    logger.info(f"Downloading model from {model_url}")
    success = download_file(model_url, model_path)
    
    if success:
        logger.info(f"Model downloaded successfully to {model_path}")
        logger.info("You can now run the Multi-Agent System with:")
        logger.info("python -m mas.langgraph_system")
    else:
        logger.error("Failed to download model")
        sys.exit(1)

if __name__ == "__main__":
    main() 