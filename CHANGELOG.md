# Changelog

All notable changes to CARE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CHANGELOG.md` file to track changes in the project.
- `CONTRIBUTING.md` file to provide guidelines for contributing to the project.
- setters/getters for energy properties of `care.Intermediate` and `care.ElementaryReaction` classes, enabling direct setting of energy properties for both species and reactions (e.g., from DFT).
- `AdsorbedSpecies`, `GasSpecies`, and `ActiveSite` child classes of `care.Intermediate` class.
- `MKMRun` dataclass for storing MKM results.
- direct evaluation of apparent activation energy of global reactions `eapp` in `DifferentialPFR.run()` method.
- direct evaluation of apparent reaction orders of reactants `napp` in `DifferentialPFR.run()` method.
- direct evaluation of degree of rate control of elementary reactions `drc` in `DifferentialPFR.run()` method.
- Orb-v3 MLIP support by adapting `care.evaluators.orb.OrbEvaluator` interface to orb-models v0.7.0.
- Color legend for CRN visualization with `care.crn.visualize.plot_crn` function.
- `cov0` argument to the MKM run function `DifferentialPFR.run()`, enabling to set the initial coverage population of the catalyst surface (before always assumed empty surface at time zero).


### Changed

- Deleted `run_kinetic` method from `ReactionNetwork` class, and rewritten `DifferentialPFR` reactor class accepting `ReactionNetwork` instances as input, instead of a list of matrix/vectors representing the network.
- `DifferentialPFR.run()` method to run MKM simulations, returning a `MKMRun` dataclass storing all output results.
- Energy propreties of `care.Intermediate` and `care.ElementaryReaction` are changed to float types and not anymore tuple with 2 elements for energy and uncertainty values. This allows for easier manipulation, especially when using MLIPs. This feature was mainly present for GAME-Net-UQ.
- Merged `care.evaluators.IntermediateEnergyEvaluator` and `care.evaluators.ReactionEnergyEvaluator` into a single `care.evaluators.UniversalEvaluator` class, which evaluates both intermediates and reactions. This simplifies the codebase and reduces redundancy.
- NEB initial and final structure generation of elementary reactions now incorporated into elementary reaction template classes.
- `ReactionNetwork` now has the `get_route_stoichiometry` method to get the stoichiometry number for all elementary reaction wrt a global reaction.
- Bump optional MLIP Sevenn from v0.11.0 to v0.13.0.
- Bump optional MLIP Orb from v0.4.2 to v0.7.0.
- `care.crn.utils.graph.atoms_to_data` function renamed to `care.crn.utils.graph.atoms_to_graph` (`data` was referring to the PyG `Data` class for representing graphs).


## [0.8.0] - 2026-07-06

### Added

- S, Br, Cl, F to the list of supported elements for CRN generation.
- Property `ReactionNetwork.global_reactions` to get list of all global reactions represented by the CRN.
- Class `GlobalReaction` to represent a global reaction in the CRN.
- Link to Google Colab in the README.md for easy testing of CARE functionalities in notebook `care_demo.ipynb`.

### Removed 

- functions from `care.crn.utils.species.py` module that were not used in the project.

## [0.7.0] - 2026-06-24

### Added

- dependency `autoadsorbate` for ACAT.
- `ReactionNetwork` factory methods `from_chemical_space`, `from_cutoffs`, and `from_species` to generate CRNs from chemical space, carbon and oxygen cutoffs, and reactants/products, respectively.

### Fixed

- Type annotation in `care.evaluators.GameNetUQ` creating import errors to the rest of the code when `gamenetuq` optional dependency was not installed.
- Reference energy of the catalyst surface when MLIPs are used, in the case of large adsorbates requiring extension of the catalyst slab.
- ACAT-based active site detection, now relying on `pymatgen` AdsorbateSiteFinder, which is more robust and reliable.

### Changed

- Upgrade dependencies: `mp-api>=0.46.2`, `pymatgen>=2025.4.10`.

### Removed

- `care.Intermediate.gas_configs` attribute.

## [0.6.0] - 2026-05-05

### Added

- Interface to UMA MLIP via the new `care.evaluators.FairChemV2IntermediateEvaluator` class.
- `connectivity`:bool item in `Intermediate.ads_configs` values to explicitly tag the final outcome of the structural relaxation in terms of correct adsorbate connectivity.
- relaxation guardrails for all MLIPs for scenarios where relaxation of adsorbates leads to dissociation. If the structure dissociates, the lowest-energy configuration after `num_configs * patience` attempts will be kept.
- Optional dependencies for `fairchemv1`, `fairchemv2`, `mace`, `orb`, `upet`, `sevennet` MLIPs, and `gamenetuq`, which can be installed with `pip install care-crn[<mlp>]`.

### Fixed

- Integration between Python and Julia for microkinetic simulations, `juliacall` now handles the entire setup automatically.


### Changed

- From static to dynamic versioning with `setuptools_scm`.
- Shift to Modular Environments: One ML evaluator, one environment. This allows for more flexibility and avoids dependency conflicts between different MLIPs.
- `care.evaluators.OCPIntermediateEvaluator` renamed to `care.evaluators.FairChemV1IntermediateEvaluator` for consistency with the new `FairChemV2IntermediateEvaluator` class.
- `care.evaluators.PETMADIntermediateEvaluator` renamed to `care.evaluators.UPETIntermediateEvaluator`.
- GAME-Net-UQ now is an optional dependency of CARE, aligning it with all other ML energy evaluators (pip install care-crn[gamenetuq]).
- `numpy` dependency upgraded to `numpy<=2.3.5`.
- `rdkit` dependency upgraded to `rdkit==2025.9.1`.
- `ase` dependency upgraded to `ase==3.26.0`. Adapted evaluator interfaces removing key `converged` from stored information for each relaxation.
- `acat` dependency upgraded to `acat==2.0.3`.
- `mp-api` dependency upgraded to `mp-api==0.46.0`.
- `pandas` dependency upgraded to `pandas==2.3.3`.
- `pydot` dependency upgraded to `pydot==4.0.1`.
- `DifferentialEquations.jl` Julia dependency upgraded to `DifferentialEquations.jl==8.0.0`.


### Removed

- dependency `torch`, `torch-geometric`. Now these come directly from the optional dependencies of the MLIPs, and are not required by CARE anymore.

## [0.5.0] - 2026-03-16

### Changed

- `care.crn.visualize.write_dotgraph` function renamed to `care.crn.visualize.plot_crn`.
- `care.ReactionNetwork.get_reaction_table` function now has an optional bool argument `return_df` to return a pandas DataFrame instead of a string. 

### Removed

- Old modules never integrated with CARE functionalities.

## [0.4.0] - 2026-02-13

### Added

- `care.io` sub-package with JSON serializers for `ReactionNetwork`, `ElementaryReaction`, `Intermediate`, and `Surface` classes.

### Changed

- CRN blueprints directly returned as `ReactionNetwork` objects instead of tuple of dict of  `Intermediate` and list of `ElementaryReaction` instances.
- Improved CRN visualization function `care.crn.visualize.write_dotgraph`.
- `care.evaluators.utils` module depending mainly on `networkx`, and not anymore on `torch-geometric`.

## [0.3.0] - 2025-12-05

### Added

- Automated evaluation of selectivity, conversion, and yield from microkinetic simulations.
- Automated Excel report generation of microkinetic simulations results.
- dependency `openpyxl` for Excel report generation.

### Changed

- Polished `care.adsorption.place_adsorbate` function.

## [0.2.0] - 2025-11-20

### Added

- Extend `care.adsorption.place_adsorbate` function for targeted adsorbate placements on user-defined surface sites and adsorbate anchoring atoms.
- `fixed_atoms` property to `care.Surface`. Before, if a surface was created from an user-provided POSCAR with some freezed atoms, the same constraint was not kept after adsorbate placement, but bottom 50% of slab atoms were freezed by default.
- Possibility to create `care.Intermediate` objects from user-defined VASP POSCAR files.

## [0.1.3] - 2025-11-06

### Added

- Installation instructions on README.md.

### Fixed

- Bug related to optional dependency `pynanoflann` for Orb MLIP.

## [0.1.2] - 2025-11-06

### Fixed

- Bug related to optional dependency `pynanoflann` for Orb MLIP.

## [0.1.1] - 2025-11-06

### Fixed

- Bug related to PyPI dependency compliance `energydiagram`.

[Unreleased]: https://github.com/LopezGroup-ICIQ/care/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/LopezGroup-ICIQ/care/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/LopezGroup-ICIQ/care/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/LopezGroup-ICIQ/care/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/LopezGroup-ICIQ/care/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/LopezGroup-ICIQ/care/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/LopezGroup-ICIQ/care/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/LopezGroup-ICIQ/care/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/LopezGroup-ICIQ/care/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/LopezGroup-ICIQ/care/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/LopezGroup-ICIQ/care/releases/tag/v0.1.1


