#!/bin/bash


# Create the conda environment from environment.yml
conda env create -f environment.yml

# Activate the newly created environment
conda activate $ENV_NAME

# Install any additional pip packages if needed
pip install acat  
pip install pubchempy
pip install rdkit
pip install git+https://github.com/giacomomarchioro/PyEnergyDiagrams

pip install -e .

# Notify the user that the environment setup is complete
echo "Conda environment '$ENV_NAME' is set up and activated."

