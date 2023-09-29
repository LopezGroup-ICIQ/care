# care-AI

This is the repository of CARE, a framework for the automatic reaction mechanism generation of heterogeneous catalytic reactions with uncertainty-based energy estimators.
It consists of 3 building blocks: 
 - R-Net, for the automatic construction and interactive reduction of reaction mechanisms for both thermal and electro heterogeneous catalytic processes.
 - GAME-Net, a graph neural networks (GNNs) for the fast estimation of the adsorption energy of molecules (containing C, H, O, N, S) on 14 transition metal surfaces.
 - pyMKM, a microkinetic modeling tool for representing experimental reactors.


## Installation

### OS/Linux

1. Clone the repo typing in the command prompt:

```bash
git clone git@github.com:LopezGroup-ICIQ/care.git
```

2. Enter the repo and create the conda environment from the .yml file

```bash
cd care
conda env create -f environment.yml
```

3. Activate the environment and install GAMERNet in editable mode

```bash
conda activate care
pip install -e .
```

### Windows

TODO 

## How to use it

Provide Jupyter notebooks

## License

The code is released under a [MIT]() license.

## References

ChemRxiv: 