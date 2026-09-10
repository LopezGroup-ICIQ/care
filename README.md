[![PyPI version](https://img.shields.io/pypi/v/care-crn.svg)](https://pypi.org/project/care-crn/)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs44286--026--00361--8-blue)](https://doi.org/10.1038/s44286-026-00361-8)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
[![Python package](https://github.com/LopezGroup-ICIQ/care/actions/workflows/python-package.yml/badge.svg)](https://github.com/LopezGroup-ICIQ/care/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/LopezGroup-ICIQ/care/graph/badge.svg)](https://codecov.io/gh/LopezGroup-ICIQ/care)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/care-crn?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/care-crn)
[![GitHub last commit](https://img.shields.io/github/last-commit/LopezGroup-ICIQ/care)](https://github.com/LopezGroup-ICIQ/care/commits/main)
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LopezGroup-ICIQ/care/blob/main/notebooks/care_demo.ipynb)

# CARE: Catalysis Automated Reaction Evaluator

<div style="display: flex; justify-content: center; align-items: center;">
    <p align="center">
     <img src="https://raw.githubusercontent.com/LopezGroup-ICIQ/care/main/care_readme_figure.png" width="80%" height="80%" />
    </p>
</div>

CARE (*Catalytic Automated Reaction Evaluator*) is a framework for the automated generation and manipulation of chemical reaction networks (CRNs) in heterogeneous catalysis. CARE is powered by ML-based energy evaluators ([GAME-Net-UQ](https://github.com/LopezGroup-ICIQ/gamenet_uq), [FairChem](https://github.com/FAIR-Chem/fairchem), [MACE](https://github.com/ACEsuit/mace), [UPET](https://github.com/lab-cosmo/pet-mad), [Orb](https://github.com/orbital-materials/orb-models), [SevenNet](https://github.com/MDIL-SNU/SevenNet)) and includes multiscale kinetic functionalities enabling the quantification of catalytic activity for reactions containing thousands of elementary steps.

## 🪛 Installation

To install CARE with a specific Machine Learning Interatomic Potential (MLIP) evaluator, specify it as an extra dependency. 

```bash
pip install "care-crn[mlip]"
```

Replace `[mlip]` with your desired evaluator. Supported models include:
`fairchemv1` | `fairchemv2` | `mace` | `upet` | `orb` | `sevenn` | `gamenetuq`

> [!WARNING]
> **Environment Isolation**
> Because each ML model depends on highly specific backend versions (PyTorch, e3nn, ASE, etc.), you will likely need to create **one distinct virtual environment** for each ML evaluator you intend to use to avoid dependency conflicts.

> [!NOTE]
> **Automatic Julia Backend Setup**
> CARE relies on a [Julia](https://julialang.org/) backend for high-performance ODE integration during microkinetic simulations. **No manual installation of Julia is required.** 
> 
> The first time you execute a simulation, `juliapkg` will automatically download a private, compatible version of Julia (if not present) and install the necessary dependencies defined in `src/care/juliapkg.json` into an isolated environment. This initial setup will take a few extra minutes to compile.


### Developer Installation

Required disk space: \~6.5 GB (Python environment), \~4.3 GB (Julia+dependencies)

1.  Clone the repo:

    ```bash
    git clone git@github.com:LopezGroup-ICIQ/care.git
    cd care
    ```

2.  Create environment:

    ```bash
    conda create -n care_env python==3.12
    conda activate care_env
    ```

3.  Install care-crn in editable mode:

    ```bash
    python3 -m pip install -e .[gamenetuq,mace,etc.]
    ```

## 💥 Usage

### Network Generation

You can construct a reaction network blueprint using one of three primary methods:
1. **Reactants and Products:** Define the start and end points using SMILES strings.
2. **Cutoffs:** Define the maximum number of Carbon (`ncc`) and Oxygen (`noc`) atoms allowed.
3. **Chemical Space:** Provide a specific list of target SMILES strings to explore (e.g., a specific decomposition network).

> [!TIP]
> **Supported Chemical Space**
> CARE currently supports reaction networks with species containing CHONS + Halogens.

```python
from care import ReactionNetwork

# Method 1: From explicit reactants and products (e.g., CO2 hydrogenation to Methanol)
crn = ReactionNetwork.from_species(reactants=["O=C=O", "[H][H]"], products=["CO", "O"])

# Method 2: From elemental network cutoffs (e.g., max 2 Carbons, 1 Oxygen)
crn = ReactionNetwork.from_cutoffs(ncc=2, noc=1)

# Method 3: From a predefined chemical space (e.g., Ethanol decomposition)
crn = ReactionNetwork.from_chemical_space(cs=["CCO"])

# Prints a quick summary (number of species, reactions, etc.)
print(crn) 

# Returns a pandas DataFrame of the generated reactions
df_reactions = crn.get_reaction_table() 

# Renders a visual graph of the reaction network
crn.plot() 
```

### Energy Evaluation

The range of catalyst materials on which CRNs can be evaluated depends on the training domain of the employed ML model.

> [!NOTE]
> **Available Evaluators**
> A complete list of available ML evaluators (and their specific capabilities) can be found in the [Evaluators README](./src/care/evaluators/README.md).

```python
from care import Surface 
from care.evaluators import MACEevaluator

# 1. Define the catalyst surface (e.g., Pt(110) from the Materials Project)
surface = Surface.from_mp("mp-2", mp_api_key="your_key", hkl="110", xy_repeat=2)
crn.add_catalyst(surface)

# 2. Initialize the ML evaluator (using GPU acceleration if available)
ml_evaluator = MACEevaluator(device="cuda", num_configs=3, max_steps=50, fmax=0.05)

# 3. Relax all adsorbed intermediates
for intermediate in crn.intermediates.values():
    ml_evaluator(intermediate)

# 4. Perform Nudged Elastic Band (NEB) for transition state searches
for reaction in crn.reactions:
    ml_evaluator(reaction, num_images=3)

# View the updated table containing the newly computed thermodynamics and barriers
crn.get_reaction_table()
```

### Microkinetic Run

Once the CRN is energetically evaluated, microkinetic simulations enable you to quantify the performance of the catalyst. CARE automatically calculates apparent kinetics ($E_{app}$, $n_{app}$) and performs robust sensitivity analyses, including the Degree of Rate Control ($\chi_{RC}$) and Degree of Selectivity Control ($\chi_{SC}$).

```python
from care.reactors import DifferentialPFR

# Initialize the reactor model
reactor = DifferentialPFR(crn)

# Define operating conditions
reactor.T = 473  # Temperature in K
reactor.P = 1e6  # Pressure in Pa
y0 = {"CO2": 0.2, "H2": 0.4, "Ar": 0.4}  # Inlet gas molar fractions
cov0 = {"H": 0.9}                        # Initial surface coverages

# Execute the simulation (automatically delegates to the Julia backend)
mkm = reactor.run(iv=y0, cov0=cov0, eapp=True, napp=True, drc=True)

# View results directly in the terminal
print(mkm.get_performance_summary())

# Export comprehensive results to Excel
mkm.export("mkm_report.xlsx")
```

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
- [CARE tutorial](./notebooks/care_demo.ipynb)  (run it on Google Colab by clicking the badge on top of the README) <br/>
- [Adsorbate placement](./notebooks/adsorbate_placement.ipynb)

## ✒️ License

The code is released under the [MIT](./LICENSE) license.

## 📜 Reference

If you use CARE in your research, please cite the following paper and consider starring the repository.

Morandi, S., Loveday, O., Renningholtz, T. *et al.* An end-to-end framework for reactivity in heterogeneous catalysis. *Nat. Chem. Eng.* (2026). [https://doi.org/10.1038/s44286-026-00361-8](https://doi.org/10.1038/s44286-026-00361-8)

```bibtex
@article{CARE,
  title = {An End-to-End Framework for Reactivity in Heterogeneous Catalysis},
  author = {Morandi, Santiago and Loveday, Oliver and Renningholtz, Tim and {Pablo-Garc{\'i}a}, Sergio and {Vargas-Hern{\'a}ndez}, Rodrigo A. and Seemakurthi, Ranga Rohit and Sanz Berman, Pol and {Garc{\'i}a-Muelas}, Rodrigo and {Aspuru-Guzik}, Al{\'a}n and L{\'o}pez, N{\'u}ria},
  year = {2026},
  journal = {Nature Chemical Engineering},
  volume = {3},
  number = {3},
  pages = {169--180},
  doi = {10.1038/s44286-026-00361-8},
}
```
