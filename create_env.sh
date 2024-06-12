# deps from conda channels

conda env create -f ENV.yml
conda activate CARE

echo `which python`
echo `which python3`

# deps from github

pip install git+https://github.com/giacomomarchioro/PyEnergyDiagrams

# test

pip install flake8 pytest

# Install Julia and related dependencies

# TODO

# end

echo "Conda environment 'CARE' is set up and activated."


