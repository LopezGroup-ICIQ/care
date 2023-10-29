#!/bin/bash


# Create the conda environment from environment.yml
conda env create -n care -f environment.yml

# Activate the newly created environment
conda init bash
conda activate care

# Install any additional pip packages if needed
pip install acat  
pip install pubchempy
pip install rdkit
pip install flake8
pip install pytest
pip install git+https://github.com/giacomomarchioro/PyEnergyDiagrams

pip install -e .

# Notify the user that the environment setup is complete
echo "Conda environment 'care' is set up and activated."

