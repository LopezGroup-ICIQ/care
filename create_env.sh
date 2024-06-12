# deps from conda channels

conda env create -f environment.yml
conda activate CARE

# deps from github

pip install git+https://github.com/giacomomarchioro/PyEnergyDiagrams

# test

pip install flake8 pytest

# Install Julia and related dependencies
curl -fsSL https://install.julialang.org | sh

julia install_jl_deps.jl

# end

echo "Conda environment 'CARE' is set up and activated."


