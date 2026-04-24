# Evaluators

This folder contains the available interfaces to the energy evaluators that can be used to evaluate the CRNs obtained with CARE. For the moment. Besides GAME-Net-UQ, you can use MACE, fairchem (both v1 or v2), Orb-v2, UPET and SevenNet machine learning interatomic potentials (MLIPs). 

Important notes:
1) The current interface implementation for these external models is experimental, you have to install the dependencies declared in their original GitHub repos (CARE README does not include those deps). 

2) Transition state evaluation can be done via MLIP-powered NEB simulations. Alternatively, barrieless reaction properties can be evaluated. 

We provide a [template](./template) folder where you can find the basic classes needed to implement your own evaluator, one interface for evaluating the species and one for the reaction properties.

# Available interfaces

| Model   | Type  | Target | Unit | Transition State   | Note |
|------------|------------|------------|------------|------------|------------|
| GAME-Net-UQ | GNN | DFT scaled adsorption energy |eV| ✅ | Direct approach
| Fairchem-v1| MLIP |DFT adsorption energy |eV|  ✅  | Structural relaxation |
| Fairchem-v2| MLIP |DFT adsorption energy |eV|  ✅  | Structural relaxation |
| MACE | MLIP | DFT total energy |eV| ✅ | Structural relaxation ||
| UPET | MLIP | DFT total energy |eV| ✅ | Structural relaxation ||
| Orb-v2 | MLIP | DFT total energy |eV| ✅ | Structural relaxation ||
| SevenNet | MLIP | DFT total energy |eV| ✅ | Structural relaxation, parallel execution not supported|
