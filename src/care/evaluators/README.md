# Evaluators

This folder contains the available interfaces to the energy evaluators that can be used to evaluate the CRNs obtained with CARE. For the moment. Besides GAME-Net-UQ, we are implementing interfaces to Open Catalyst OC20 models and MACE-MP-0 (work in progress). 

Important notes:
1) The current interface implementation for these external models is experimental, you have to install the dependencies declared in their original GitHub repos (CARE README does not include those deps). 

2) These new interfaces focus on structural relaxation and do not cover TS evaluation (for now). Feel free to contribute, in the meanwhile, reaction properties are directly evaluated from the intermediates (barrier-less steps, no lateral interactions). 

We provide a [template](./template) folder where you can find the basic classes needed to implement your own evaluator, one interface for evaluating the species and one for the reaction properties.

# Available interfaces

| Model   | Type  | Target | Unit | Transition State   | Note |
|------------|------------|------------|------------|------------|------------|
| GAME-Net-UQ | GNN | DFT scaled adsorption energy |eV| ✅ (bond-breaking steps) | Direct approach
| OC20 models| MLIP |DFT adsorption energy |eV|  	❌️ (feasible!) | Structural relaxation |
| MACE-MP-0 | MLIP | DFT total energy |eV|  	❌️ (feasible!) | Structural relaxation ||

