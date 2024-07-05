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
 
 #

CARE (*Catalysis Automated Reaction Evaluator*) is a tool for generating and manipulating chemical reaction networks (CRNs) for catalysis on transition metal surfaces. CARE is powered with GAME-Net-UQ, a graph neural network with uncertainty quantification (UQ) targeting the DFT energy of relaxed species and transition states.

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
python3 -m pip install . # Add '-e' before '.' to install care in development mode 
```

3. Install [Julia](https://julialang.org/) to run microkinetic simulations with it. This step is required if you want to perform microkinetic simulations with Julia. As alternative, simulations can run with the implemented Scipy solver.

```bash
curl -fsSL https://install.julialang.org | sh
```

4. Install the required Julia dependencies creating the Julia environment provided by `Project.toml`

```bash
TODO
```

## Usage

The current way to generate CRNs in CARE requires setting up a .toml configuration file and running the code with:

```bash
care_run -i input.toml -o output_name
```

This will generate a directory `output_name` containing a `crn.pkl` with the generated reaction network.
Details about setting up the configuration file are found in ``.

Further details will be uploaded soon.

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

## License

The code is released under the [MIT](./LICENSE) license.

## Reference

- **A Foundational Model for Reaction Networks on Metal Surfaces**  
  Authors: S. Morandi, O. Loveday, T. Renningholtz, S. Pablo-García, R. A. Vargas Hernáńdez, R. R. Seemakurthi, P. Sanz Berman, R. García-Muelas, A. Aspuru-Guzik, and N. López  
  DOI: [10.26434/chemrxiv-2024-bfv3d](https://doi.org/10.26434/chemrxiv-2024-bfv3d)
