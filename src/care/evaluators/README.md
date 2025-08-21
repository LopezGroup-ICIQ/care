# Evaluators

This folder contains the available interfaces to the energy evaluators that can be used to evaluate the CRNs obtained with CARE. For the moment. Besides GAME-Net-UQ, you can use MACE, fairchem, Orb, PET-MAD and SevenNet machine learning potentials (MLPs). 

Important notes:
1) The current interface implementation for these external models is experimental, you have to install the dependencies declared in their original GitHub repos (CARE README does not include those deps). 

2) Transition state evaluation can be done via MLP-powered NEB simulations. Alternatively, barrieless reaction properties can be evaluated. 

We provide a [template](./template) folder where you can find the basic classes needed to implement your own evaluator, one interface for evaluating the species and one for the reaction properties.

# Available interfaces

| Model   | Type  | Target | Unit | Transition State   | Note |
|------------|------------|------------|------------|------------|------------|
| GAME-Net-UQ | GNN | DFT scaled adsorption energy |eV| ✅ | Direct approach
| Fairchem OC models| MLIP |DFT adsorption energy |eV|  ✅  | Structural relaxation |
| MACE | MLP | DFT total energy |eV| ✅ | Structural relaxation ||
| PET-MAD | MLP | DFT total energy |eV| ✅ | Structural relaxation ||
| Orb | MLP | DFT total energy |eV| ✅ | Structural relaxation ||
| SevenNet | MLP | DFT total energy |eV| ✅ | Structural relaxation, parallel execution not supported|
