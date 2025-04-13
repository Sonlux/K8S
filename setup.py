from setuptools import setup, find_packages
import subprocess
import sys
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

def install_requirements():
    """Install requirements from requirements.txt"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed all requirements!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # First install requirements
    install_requirements()
    
    # Then setup the package
    setup(
        name="mas",
        version="0.1.0",
        author="Your Name",
        author_email="your.email@example.com",
        description="Multi-Agent System for Kubernetes Monitoring and Remediation",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/yourusername/mas",
        packages=find_packages(),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        python_requires=">=3.8",
        install_requires=requirements,
        entry_points={
            "console_scripts": [
                "mas=mas.__main__:main",
            ],
        },
    ) 