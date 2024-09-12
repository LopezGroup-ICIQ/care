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

CARE (*Catalysis Automated Reaction Evaluator*) is a tool for generating and manipulating chemical reaction networks (CRNs) on catalytic surfaces. CARE is powered with [GAME-Net-UQ](https://github.com/LopezGroup-ICIQ/gamenet_uq), a graph neural network with uncertainty quantification targeting the DFT energy of relaxed species and transition states.

## 🪛 Installation

Installing CARE requires Conda and Git locally installed. The following instructions are optimized to install CARE on Linux systems, while for macOS we noticed a lower performance in the CRN generation mainly due to Python multiprocessing (see *Contexts and start methods* in the [documentation](https://docs.python.org/3/library/multiprocessing.html))

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
conda install pytorch cpuonly pytorch-scatter pytorch-sparse pyg -c pytorch -c pyg
```

*NOTE: MacOS users might need to install pytorch geometric using pip.*

5. (optional) Install [Julia](https://julialang.org/) and the ODE packages required to perform microkinetic simulations. As alternative, simulations can run with the implemented Scipy solver.

```bash
curl -fsSL https://install.julialang.org | sh
python3 -m pip install juliacall  # Python-Julia bridge
julia -e 'import Pkg; Pkg.add("DifferentialEquations"); Pkg.add("DiffEqGPU"); Pkg.add("CUDA");'
```
*NOTE: For some systems Julia may present some error while using sh. If that is the case, please install Julia by running instead:*

```bash
curl -fsSL https://install.julialang.org | sh -s -- -y
```

6. (optional) Install the different evaluators available ([MACE](https://github.com/ACEsuit/mace), [fairchem](https://github.com/FAIR-Chem/fairchem)), through the following command:

*NOTE: There currently is a dependency clash during installation for the two evaluators related to the `e3nn` library (see: [this issue for MACE](https://github.com/ACEsuit/mace/issues/555)). Installation might result in an incompatibility warning, but
both libraries should work correctly if the installation order shown below is followed.*

```bash
python3 -m pip install fairchem-core
python3 -m pip install mace-torch
```

## 💥 Usage

### Blueprint generation

```bash
gen_crn_blueprint -h  # documentation
gen_crn_blueprint [-ncc NCC] [-noc NOC] [-electro ELECTRO] [-o OUTPUT] [-ncpu NUM_CPU]
```

This script does not require any input file. The output is stored as pickle file. To access the blueprint, do:

```python
from pickle import load

with open('path_to_blueprint_file', 'rb') as f:
    intermediates, reactions = load(f)
```

### Evaluation of intermediate and reaction properties

```bash
eval_crn -h  # documentation
eval_crn [-i INPUT] [-bp BP] [-o OUTPUT] [-ncpu NUM_CPU]
```

This script requires an input toml file defining the surface of interest, property evaluators, and their settings. The output is a ``ReactionNetwork`` object stored as pickle file. You can find an example input file [here](./src/care/scripts/example_eval.toml).

### Microkinetic simulation

```bash
run_kinetic [-i INPUT] [-crn CRN] [-o OUTPUT]
```

This script runs microkinetic simulation starting from the evaluated reaction network and an input toml file defining the reaction conditions, solver, inlet conditions. The results are stored as a pickle object file.

### Run all together

You can run the entire pipeline (blueprint generation -> properties evaluation -> kinetic simulation) running the `care_run` script:

```bash
care_run -h  # documentation
care_run -i input.toml -o output_name
```

This will generate a directory `output_name` containing a `crn.pkl` with the generated reaction network.
Examples of input .toml files can be found in `src/care/scripts` and `src/care/examples`.

## 📖 Tutorials

We currently provide three tutorials, available in the ``notebooks`` directory:
- [CRN generation and manipulation](./notebooks/care_demo.ipynb) <br/>
- [Energy evaluator interface implementation](./notebooks/interface_demo.ipynb) <br/>
- [Microkinetic simulations](./notebooks/kinetics_demo.ipynb)

## ❗️Notes

The DFT database in ASE format used to retrieve available CRN intermediates will be uploaded soon in Zenodo.

## ✒️ License

The code is released under the [MIT](./LICENSE) license.

## 📜 Reference

- **A Foundational Model for Reaction Networks on Metal Surfaces**
  Authors: S. Morandi, O. Loveday, T. Renningholtz, S. Pablo-García, R. A. Vargas Hernáńdez, R. R. Seemakurthi, P. Sanz Berman, R. García-Muelas, A. Aspuru-Guzik, and N. López
  DOI: [10.26434/chemrxiv-2024-bfv3d](https://doi.org/10.26434/chemrxiv-2024-bfv3d)
