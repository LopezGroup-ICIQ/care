[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![Python package](https://github.com/LopezGroup-ICIQ/care/actions/workflows/python-package.yml/badge.svg)](https://github.com/LopezGroup-ICIQ/care/actions/workflows/python-package.yml)
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)

# CARE: Catalysis Automated Reaction Evaluator

<div style="display: flex; justify-content: center; align-items: center;">
    <p align="center">
     <img src="./CARE_github.png" width="90%" height="90%" />
    </p>
</div>

CARE (*Catalysis Automated Reaction Evaluator*) is a tool for generating and manipulating chemical reaction networks (CRNs) for catalysis on transition metal surfaces. CARE is powered with [GAME-Net-UQ](https://github.com/LopezGroup-ICIQ/gamenet_uq), a graph neural network with uncertainty quantification targeting the DFT energy of relaxed species and transition states.

## Installation

We recommend creating an environment with Conda to install the package.

1. Clone the repo:

```bash
git clone git@github.com:LopezGroup-ICIQ/care.git
```

2. Create a conda environment with Python 3.11 and activate it:

```bash
conda create -n care_env python=3.11
conda activate care_env
```

3. Enter the repo and install the package with pip:

```bash
cd care
python3 -m pip install . 
```

*NOTE: MacOS users might need to launch a new shell at this point in order for the entry points to work correctly.*

4. Install `pytorch` and `pytorch_geometric` through conda. **Ensure that the libraries are obtained from the pytorch and pyg channels**, as shown here:

```bash
conda install pytorch cpuonly pyg -c pytorch -c pyg
```

*NOTE: MacOS users might need to install pytorch geometric using pip.*

5. Install [Julia](https://julialang.org/). This step is required if you want to perform microkinetic simulations with Julia solvers. As alternative, simulations can run with the implemented Scipy solver.

```bash
curl -fsSL https://install.julialang.org | sh
```
*NOTE: For some systems Julia may present some error while using sh. If that is the case, please install Julia by running instead the following command.*

```bash
curl -fsSL https://install.julialang.org | sh -s -- -y
```

6. Install the required Julia dependencies by running the following:

```bash
python3 -m pip install julia
julia -e 'import Pkg; Pkg.add("PyCall"); Pkg.add("CUDA"); Pkg.add("DifferentialEquations"); Pkg.add("DiffEqGPU");'
```

These packages allow to run microkinetic simulations with Julia calling it from Python.

## Usage

The current way to generate chemical reaction networks in CARE requires setting up a .toml configuration file and running the code:

```bash
care_run -i input.toml -o output_name
```

This will generate a directory `output_name` containing a `crn.pkl` with the generated reaction network.
Examples of input .toml files can be found in `src/care/scripts/example_c1o2.toml` and `src/care/scripts/example_c2o4.toml`.

A step by step tutorial for generating and manipulating reaction networks is in progresso in ``notebooks/care_demo.ipynb``.

## How to access the CRN

```python
from pickle import load

with open('./C1O4/Pd111/crn.pkl', 'rb') as pickle_file:
    crn = load(pickle_file)
```

`crn` is a `care.ReactionNetwork` object which provides rapid access to the intermediates (`care.Intermediate`), elementary reactions (`care.ElementaryReaction`), and its properties as activation barrier `care.ElementaryReaction.e_act` and reaction energy `care.ElementaryReaction.e_rxn`.

To visualize a specific elementary step:

```python
crn.visualize_reaction(0)
```

## Notes

The DFT database in ASE format used to retrieve available CRN intermediates will be uploaded soon in Zenodo.

## License

The code is released under the [MIT](./LICENSE) license.

## Reference

- **A Foundational Model for Reaction Networks on Metal Surfaces**  
  Authors: S. Morandi, O. Loveday, T. Renningholtz, S. Pablo-García, R. A. Vargas Hernáńdez, R. R. Seemakurthi, P. Sanz Berman, R. García-Muelas, A. Aspuru-Guzik, and N. López  
  DOI: [10.26434/chemrxiv-2024-bfv3d](https://doi.org/10.26434/chemrxiv-2024-bfv3d)
