[![PyPI version](https://img.shields.io/pypi/v/care-crn.svg)](https://pypi.org/project/care-crn/)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs44286--026--00361--8-blue)](https://doi.org/10.1038/s44286-026-00361-8)
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

CARE (*Catalytic Automated Reaction Evaluator*) is a framework for the automated generation and manipulation of chemical reaction networks (CRNs) in heterogeneous catalysis. CARE is powered by ML-based energy evaluators ([GAME-Net-UQ](https://github.com/LopezGroup-ICIQ/gamenet_uq), [FairChem](https://github.com/FAIR-Chem/fairchem), [MACE](https://github.com/ACEsuit/mace) potentials, *etc*.) and includes kinetic functionalities enabling the quantification of catalytic activity for reactions containing thousands of elementary steps.

## 🪛 Installation

### 1\. From PyPI

```bash
pip install care-crn
```

### 2\. ML evaluators

`care-crn` interfaces with several external ML models, most of them ML interatomic potentials (MLIPs). These must be installed separately as they depend on different versions of Pytorch, causing conflicts. You can install [FairChemV1](https://github.com/FAIR-Chem/fairchem) or [FairChemV2](https://github.com/FAIR-Chem/fairchem), [MACE](https://github.com/ACEsuit/mace), [UPET](https://github.com/lab-cosmo/pet-mad), [Orb-v2](https://github.com/orbital-materials/orb-models), and [SevenNet](https://github.com/MDIL-SNU/SevenNet) by running:

```bash
pip install care-crn[mace]
pip install care-crn[fairchemv1]
pip install care-crn[fairchemv2]
pip install care-crn[petmad]
pip install care-crn[orb]
pip install care-crn[sevennet]
pip install care-crn[gamenetuq]
```

It is important to note that since each ML model depends on specific versions of Python packages (pytorch, e3nn, ase, etc.), starting from care-crn==0.6.0 you will need to create one distinct environment for each ML evaluator you want to employ. 


#### Julia-powered microkinetic solver

To run microkinetic simulations with [Julia](https://julialang.org/), install it and the required packages:

```bash
curl -fsSL https://install.julialang.org | sh -s -- --yes && ~/.juliaup/bin/juliaup add 1.11
julia -e 'import Pkg; Pkg.add("DifferentialEquations"); Pkg.add("LinearSolve");'
```

*⏲ Julia setup time estimate: \~13min (Ubuntu), \~9min (macOS)*

-----

### 3\. Developer Installation

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

    ```bash
    python3 -m pip install -e .
    ```

    *NOTE: macOS users might need to launch a new shell at this point in order for the entry points to work correctly.*

4.  **Install optional dependencies:**

    ```bash
    python3 -m pip install -e .[mace]
    python3 -m pip install -e .[fairchemv2]
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

CRNs in CARE are stored as compressed .json files.

```python
from care.io import load_network

crn = load_network("blueprint.json.gz")
```

### Evaluation of intermediate and reaction properties

The range of catalyst materials on which CRNs can be evaluated depends on the domain of the data-driven energy evaluator employed.
Currently, CARE provides interfaces to GAME-Net-UQ, FairChem-v1 potentials, MACE, Orb, PET-MAD, and SevenNet.

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

This will generate a `output_name` folder with the generated reaction network and additional results from the kinetic simulation.
Examples of input .toml files can be found [here](./src/care/scripts/input_examples/care_script/).

## 📖 Tutorials

We currently provide two tutorials, available in the ``notebooks`` directory:
- [CARE tutorial](./notebooks/care_demo.ipynb) <br/>
- [Adsorbate placement](./notebooks/adsorbate_placement.ipynb).

## ✒️ License

The code is released under the [MIT](./LICENSE) license.

## 📜 Reference

Morandi, S., Loveday, O., Renningholtz, T. *et al.* An end-to-end framework for reactivity in heterogeneous catalysis. *Nat. Chem. Eng.* (2026). [https://doi.org/10.1038/s44286-026-00361-8](https://doi.org/10.1038/s44286-026-00361-8)
