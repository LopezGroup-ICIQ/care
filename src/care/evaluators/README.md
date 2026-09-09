# Evaluators

This sub-library contains the interfaces to the energy evaluators that can be used to evaluate the CRNs obtained with CARE. For the moment. Besides GAME-Net-UQ, you can use MACE, Fairchem (both v1 or v2), Orb, UPET and SevenNet machine learning interatomic potentials (MLIPs). 

Important notes:
1) The current interface implementation for these external models is experimental, you have to install the dependencies declared in their original GitHub repos. 

2) Transition state search for surface steps can be done via MLIP-powered NEB simulations.

# Available interfaces

| Model   | Type  | Target | Unit | Transition State   | Note | Installation |
|------------|------------|------------|------------|------------|------------|------------|
| GAME-Net-UQ | GNN | DFT scaled adsorption energy |eV| ✅ | Direct approach | `pip install care-crn[gamenetuq]`|
| Fairchem-v1| MLIP |DFT adsorption energy |eV|  ✅  | Structural relaxation | `pip install care-crn[fairchemv1]` |
| Fairchem-v2| MLIP |DFT total energy |eV|  ✅  | Structural relaxation | `pip install care-crn[fairchemv2]` |
| MACE | MLIP | DFT total energy |eV| ✅ | Structural relaxation | `pip install care-crn[mace]` |
| UPET | MLIP | DFT total energy |eV| ✅ | Structural relaxation | `pip install care-crn[upet]` |
| Orb | MLIP | DFT total energy |eV| ✅ | Structural relaxation | `pip install care-crn[orb]` |
| SevenNet | MLIP | DFT total energy |eV| ✅ | Structural relaxation| `pip install care-crn[sevenn]` |

*GNN = Graph Neural Network; MLIP = Machine Learned Interatomic Potential; DFT = Density Funtional Theory*
