#!/bin/bash

# Set the script to fail on any command error
set -e

# Define environment name and Python version variables
ENV_NAME="spoc_env2"
PYTHON_VERSION="3.11"

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found, please install it first."
    exit 1
fi

# Update APT and install dependencies
sudo apt-get update
sudo apt-get install -y git curl wget pandoc software-properties-common openjdk-11-jdk

# Create and activate Conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
conda activate $ENV_NAME

# Install Conda packages
conda install -y -c conda-forge openbabel

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
    exit 1
fi

# Install Python package from the current directory
pip install .

echo "Environment $ENV_NAME is set up and ready."
