[![PyPI version](https://img.shields.io/pypi/v/care-crn.svg)](https://pypi.org/project/care-crn/)
[![DOI](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2024--bfv3d-blue)](https://doi.org/10.26434/chemrxiv-2024-bfv3d)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.12-blue.svg)
[![Python package](https://github.com/LopezGroup-ICIQ/care/actions/workflows/python-package.yml/badge.svg)](https://github.com/LopezGroup-ICIQ/care/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/LopezGroup-ICIQ/care/graph/badge.svg)](https://codecov.io/gh/LopezGroup-ICIQ/care)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/care-crn?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/care-crn)
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)

# CARE: Catalysis Automated Reaction Evaluator

<div style="display: flex; justify-content: center; align-items: center;">
    <p align="center">
     <img src="https://raw.githubusercontent.com/LopezGroup-ICIQ/care/main/care_readme_figure.png" width="80%" height="80%" />
    </p>
</div>

CARE (*Catalysis Automated Reaction Evaluator*) is a tool for generating and manipulating chemical reaction networks (CRNs) on catalytic surfaces. CARE is powered by data-driven models such as [GAME-Net-UQ](https://github.com/LopezGroup-ICIQ/gamenet_uq), [Open Catalyst](https://github.com/FAIR-Chem/fairchem) models, [MACE](https://github.com/ACEsuit/mace), etc.

## 🪛 Installation

We recommend installing `care-crn` from PyPI. A developer installation is also available for those who wish to contribute.

### 1\. Standard Installation (from PyPI)

This is the fastest way to get `care-crn` and its core dependencies.

```bash
pip install care-crn
```

### 2\. Install External Evaluators & Runtimes

`care-crn` interfaces with several external ML Interatomic Potentials (MLIPs). These must be installed separately.

#### MLIP Evaluators (OCP, MACE, etc.)

You can install the Python wrappers for these evaluators using `pip`'s "extras" syntax.

1.  **For OCP ([FAIRChem-v1](https://github.com/FAIR-Chem/fairchem)):**
    First, install `torch_sparse` and `torch_scatter` by following the instructions on the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) page. Then, run:

    ```bash
    pip install care-crn[ocp]
    ```

2.  **For other evaluators:**
    You can install [MACE](https://github.com/ACEsuit/mace), [PET-MAD](https://github.com/lab-cosmo/pet-mad), [Orb](https://github.com/orbital-materials/orb-models), and [SevenNet](https://github.com/MDIL-SNU/SevenNet) by running:

    ```bash
    pip install care-crn[mace]
    pip install care-crn[petmad]
    pip install care-crn[orb]
    pip install care-crn[sevennet]
    ```

    Or all at once:

    ```bash
    pip install care-crn[ocp,mace,petmad,orb,sevennet]
    ```

    *NOTE: There currently is a dependency clash during installation of OCP and MACE evaluators related to the `e3nn` library (see: [this issue for MACE](https://github.com/ACEsuit/mace/issues/555)). Installation might result in an incompatibility warning, but
    both evaluators should work correctly if the installation order shown above is followed.*

#### Julia (Microkinetic modeling)

To run microkinetic simulations with [Julia](https://julialang.org/), install it and the required packages:

```bash
# Install Julia and add version 1.11
curl -fsSL https://install.julialang.org | sh -s -- --yes && ~/.juliaup/bin/juliaup add 1.11

# Add DifferentialEquations.jl and LinearSolve.jl
julia -e 'import Pkg; Pkg.add("DifferentialEquations"); Pkg.add("LinearSolve");'
```

*⏲ Julia setup time estimate: \~13min (Ubuntu), \~9min (macOS)*

-----

### 3\. (Optional) Developer Installation (from Source)

If you want to contribute to the code or use the very latest (unstable) version, you can install from the source.

  * ⏲ **Total installation time estimates:** \~18min (Ubuntu), \~11min (macOS).
  * 💾 **Required disk space:** \~6.5 GB (Conda environment), \~4.3 GB (Julia+dependencies)

<!-- end list -->

1.  **Clone the repo:**

    ```bash
    git clone git@github.com:LopezGroup-ICIQ/care.git
    cd care
    ```

2.  **Create a conda environment:**

    ```bash
    conda create -n care_env python==3.12
    conda activate care_env
    ```

3.  **Install the package in "editable" mode:**
    (The `-e` flag links the installation to your source code)

    ```bash
    python3 -m pip install -e .
    ```

    *NOTE: macOS users might need to launch a new shell at this point in order for the entry points to work correctly.*

4.  **Install optional dependencies:**
    (Note the syntax is slightly different from the PyPI install)

    ```bash
    python3 -m pip install -e .[ocp]
    python3 -m pip install -e .[mace]
    # etc.
    ```

## 💥 Usage

### Blueprint generation

The blueprint can be constructed in two ways, by providing (i) the network carbon and oxygen cutoffs *ncc* and *noc*, or (ii) the chemical space as list of SMILES.

```bash
gen_crn_blueprint -h  # documentation
gen_crn_blueprint -ncc 2 -noc 1 -o output_name  # Example from ncc and noc
gen_crn_blueprint -cs "CCO" "C(CO)O" -o output_name # Example from user-defined chemical space
```

<div style="display: flex; justify-content: center; align-items: center;">
    <p align="center">
     <img src="https://raw.githubusercontent.com/LopezGroup-ICIQ/care/main/care_bp_screenshot.png" width="70%" height="70%" />
    </p>
</div>

The CRN blueprint is stored as pickle file. To access the blueprint, do:

```python
from pickle import load

with open('path_to_blueprint_file', 'rb') as f:
    intermediates, reactions = load(f)
```

### Evaluation of intermediate and reaction properties

The range of catalyst materials on which CRNs can be constructed depends on the domain of the data-driven energy evaluator employed to derive the reaction properties.
Currently, CARE provides interfaces to GAME-Net-UQ, OCP models, and MACE-MP potentials.

```bash
eval_crn -h  # documentation
eval_crn [-i INPUT] [-bp BP] [-o OUTPUT] [-ncpu NUM_CPU]
```

This script requires an input toml file defining the material/surface of interest, the model of choice and its settings. The output is a ``ReactionNetwork`` object stored as pickle file. You can find examples of input files [here](./src/care/scripts/input_examples/eval_crn/). 

For macOS we noticed a lower performance in the CRN generation due to Python multiprocessing (see *Contexts and start methods* in the [documentation](https://docs.python.org/3/library/multiprocessing.html))


### Microkinetic simulation

```bash
run_kinetic [-i INPUT] [-crn CRN] [-o OUTPUT]
```

This script runs microkinetic simulation starting from the evaluated reaction network and an input toml file defining the reaction conditions, solver, inlet conditions. The results are stored as a pickle object file.

### Run all together

You can run the entire pipeline (blueprint generation ➡ energy evaluation ➡ kinetic simulation) running the `care_run` script:

```bash
care_run -h  # documentation
care_run -i input.toml -o output_name
```

This will generate a directory `output_name` containing a `crn.pkl` with the generated reaction network.
Examples of input .toml files can be found [here](./src/care/scripts/input_examples/care_script/).

## 📖 Tutorials

We currently provide three tutorials, available in the ``notebooks`` directory:
- [CRN generation and manipulation](./notebooks/care_demo.ipynb) <br/>
- [Energy evaluator interface implementation](./notebooks/interface_demo.ipynb) <br/>
- [Microkinetic simulations](./notebooks/kinetics_demo.ipynb) <br/>
- [Adsorbate placement](./notebooks/adsorbate_placement.ipynb)

## ✒️ License

The code is released under the [MIT](./LICENSE) license.

## 📜 Reference

- **A Foundational Model for Reaction Networks on Metal Surfaces**
  Authors: S. Morandi, O. Loveday, T. Renningholtz, S. Pablo-García, R. A. Vargas Hernáńdez, R. R. Seemakurthi, P. Sanz Berman, R. García-Muelas, A. Aspuru-Guzik, and N. López
  DOI: [10.26434/chemrxiv-2024-bfv3d](https://doi.org/10.26434/chemrxiv-2024-bfv3d)
