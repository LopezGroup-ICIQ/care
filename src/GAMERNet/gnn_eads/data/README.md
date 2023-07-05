# Data directory

Here all the data used in this work are stored. 

- `FG_dataset` stands for "functional groups" dataset. It stores the DFT training data grouped by chemical family in different sub-folders (e.g., `oximes`). Each sub-folder contains at least these two files:  
    1. `structures`: Stores the VASP CONTCAR geometry files of that specific chemical family, named following the convention `xx-1234-a.contcar`, where `xx` refers to the metal symbol in lowercase, each digit in `123` refers to the number of C, H, O (this last one is the default, if not S, N, or combinations) atoms in the adsorbate, `4` is used to distinguish isomers. The last character refers to the adsorption configuration. Examples: ethylene on platinum is `pt-2401-a`, formaldehyde on gold is `au-1211-a`, aniline on rhodium is `rh-67N1-a`.
    2. `energies.dat`: A file containing the ground state energy in eV of each sample. Each line represents a sample. Example: `ag-37X3-a -193.89207256`.

- `BM_dataset` includes "big molecules" of industrial interest (biomass, plastics and polyurethanes). It is used for testing the GNN performance in extrapolation mode. For each family of molecules, 5 representative molecules have been considered adsorbed on two metal catalysts, based on the specific application.