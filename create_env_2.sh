#!/bin/bash


# Create the conda environment from environment.yml
conda env create -n care -f test_care.yml

# Activate the newly created environment
conda init bash
conda activate care

pip install -e .

# Notify the user that the environment setup is complete
echo "Conda environment 'care' is set up and activated."

