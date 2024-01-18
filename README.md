[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![Python package](https://github.com/LopezGroup-ICIQ/care/actions/workflows/python-package.yml/badge.svg)](https://github.com/LopezGroup-ICIQ/care/actions/workflows/python-package.yml)
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)


# CARE: Catalysis Automatic Reaction Evaluator

<div style="display: flex; justify-content: center; align-items: center;">
    <p align="center">
     <img src="./output.gif" width="60%" height="60%" />
    </p>
</div>
 
 #

CARE (*Catalysis Automatic Reaction Evaluator*) is a package for automatic construction of chemical reaction networks (CRNs) for heterogeneous catalysis on metal surfaces. Prediction of activation and reaction energies is performed with GAME-Net UQ, a graph neural network with uncertainty quantification (UQ).

## Installation

1. Clone the repo typing in the command prompt:

```bash
git clone git@github.com:LopezGroup-ICIQ/care.git
```

2. Enter the repo and run the bash script `create_env.sh`

```bash
cd care
./create_env.sh 
```

This will automatically set up and activate the conda environment.

## How to use it

The current way to generate CRNs in CARE requires three steps:

1) **Template generation**: Given a Network Carbon Cutoff `ncc` and Network Oxygen Cutoff `noc`:

```bash
cd scripts
python netgen.py -ncc 1 -noc 3
```

This will generate a directory `C1O3` containing `intermediates.pkl` and `reactions.pkl`, defining the CRN template general for all available surfaces.

2) **Adsorbate placement**: Given a metal `m` and a specific surface orientation `hkl`, adsorb the intermediates on the surface:

```bash
python adsorb.py -i C1O3 -m Ru -hkl 0001
```

A subdirectory `C1O3/Ru0001` is then created, containing all the adsorbed intermediates. This step represents currently the bottleneck of the code, as it could take a long time depending on the size of the CRN, defined by `ncc` and `noc`.

3) **Energy evaluation**: Evaluate the activation barrier for all bond breaking/forming reactions and the reaction energy for all elementary steps:

```bash
python evaluate.py -i C1O3/Ru0001
```

Once finished, `C1O3/Ru0001/crn.pkl` is created, containing the whole CRN with energy included.

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

The code is released under a [MIT]() license.

## References

ChemRxiv: 
