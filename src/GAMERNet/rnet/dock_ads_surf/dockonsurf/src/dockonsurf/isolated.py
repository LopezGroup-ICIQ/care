"""Functions to generate the conformers to be adsorbed and the most stable one.

functions:
remove_C_linked_Hs: Removes hydrogens bonded to a carbon atom from a molecule.
gen_confs: Generate a number of conformers in random orientations.
get_moments_of_inertia: Computes moments of inertia of the given conformers.
get_moments_of_inertia: Computes the moments of inertia of the given conformers.
pre_opt_confs: Optimizes the geometry of the given conformers and returns the
    new mol object and the energies of its conformers.
run_isolated: directs the execution of functions to achieve the goal
"""
import logging

import numpy as np
import rdkit.Chem.AllChem as Chem

logger = logging.getLogger('DockOnSurf')


def remove_C_linked_Hs(mol: Chem.rdchem.Mol):
    """Removes hydrogen atoms bonded to a carbon atom from a rdkit mol object.

    @param mol: rdkit mol object of the molecule with hydrogen atoms.
    @return: rdkit mol object of the molecule without hydrogen atoms linked to
    a carbon atom.

    The functions removes the hydrogen atoms bonded to carbon atoms while
    keeping the ones bonded to other atoms or non-bonded at all.
    """

    mol = Chem.RWMol(mol)
    rev_atm_idxs = [atom.GetIdx() for atom in reversed(mol.GetAtoms())]

    for atm_idx in rev_atm_idxs:
        atom = mol.GetAtomWithIdx(atm_idx)
        if atom.GetAtomicNum() != 1:
            continue
        for neigh in atom.GetNeighbors():
            if neigh.GetAtomicNum() == 6:
                mol.RemoveAtom(atom.GetIdx())
    return mol


def gen_confs(mol: Chem.rdchem.Mol, num_confs: int):
    """Generate conformers in random orientations.

    @param mol: rdkit mol object of the molecule to be adsorbed.
    @param num_confs: number of conformers to randomly generate.
    @return: mol: rdkit mol object containing the different conformers.
             rmsd_mtx: Matrix with the rmsd values of conformers.

    Using the rdkit library, conformers are randomly generated. If structures 
    are required to be local minima, ie. setting the 'local_min' value to 
    True, a geometry optimisation using UFF is performed.
    """
    logger.debug('Generating Conformers.')

    mol = Chem.AddHs(mol)
    Chem.EmbedMultipleConfs(mol, numConfs=num_confs, numThreads=0,
                            randomSeed=5817216)
    Chem.AlignMolConformers(mol)
    logger.info(f'Generated {len(mol.GetConformers())} conformers.')
    return mol


def get_moments_of_inertia(mol: Chem.rdchem.Mol):
    """Computes the moments of inertia of the given conformers.

    @param mol: rdkit mol object of the relevant molecule.
    @return numpy array 2D: The inner array contains the moments of inertia for
    the three principal axis of a given conformer. They are ordered by its value
    in ascending order. The outer tuple loops over the conformers.
    """
    from rdkit.Chem.Descriptors3D import PMI1, PMI2, PMI3

    return np.array([[PMI(mol, confId=conf) for PMI in (PMI1, PMI2, PMI3)]
                     for conf in range(mol.GetNumConformers())])


def pre_opt_confs(mol: Chem.rdchem.Mol, force_field='mmff', max_iters=2000):
    """Optimizes the geometry of the given conformers and returns the new mol
    object and the energies of its conformers.

    @param mol: rdkit mol object of the relevant molecule.
    @param force_field: Force Field to use for the pre-optimization.
    @param max_iters: Maximum number of geometry optimization iterations. With 0
    a single point energy calculation is performed and only the conformer
    energies are returned.
    @return mol: rdkit mol object of the optimized molecule.
    @return numpy.ndarray: Array with the energies of the optimized conformers.

    The MMFF and UFF force fields can be used for the geometry optimization in
    their rdkit implementation. With max_iters value set to 0, a single point
    energy calculation is performed and only the energies are returned. For
    values larger than 0, if the geometry does not converge for a certain
    conformer, the latter is removed from the list of conformers and its energy
    is not included in the returned list.
    """
    from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMoleculeConfs, \
        UFFOptimizeMoleculeConfs

    init_num_confs = mol.GetNumConformers()
    if force_field == 'mmff':
        results = np.array(MMFFOptimizeMoleculeConfs(mol, numThreads=0,
                                                     maxIters=max_iters,
                                                     nonBondedThresh=10))
    elif force_field == 'uff':
        results = np.array(UFFOptimizeMoleculeConfs(mol, numThreads=0,
                                                    maxIters=max_iters,
                                                    vdwThresh=10))
    else:
        logger.error("force_field parameter must be 'MMFF' or 'UFF'")
        raise ValueError("force_field parameter must be 'MMFF' or 'UFF'")

    # Remove non-converged conformers if optimization is on, ie. maxIters > 0
    # return all conformers if optimization is switched off, ie. maxIters = 0
    if max_iters > 0:
        for i, conv in enumerate(results[:, 0]):
            if conv != 0:
                mol.RemoveConformer(i)
        for i, conf in enumerate(mol.GetConformers()):
            conf.SetId(i)
        if mol.GetNumConformers() < init_num_confs:
            logger.warning(f'Geometry optimization did not comverge for at'
                           f'least one conformer. Continuing with '
                           f'{mol.GetNumConformers()} converged conformers.')
        logger.info(f'Pre-optimized conformers with {force_field}.')
        return mol, np.array([res[1] for res in results if res[0] == 0])
    else:
        logger.info(f'Computed conformers energy with {force_field}.')
        return np.array([res[1] for res in results])


def run_isolated(inp_vars):
    """Directs the execution of functions to obtain the conformers to adsorb

    @param inp_vars: Calculation parameters from input file.
    @return:
    """
    from src.dockonsurf.formats import adapt_format, confs_to_mol_list, \
        rdkit_mol_to_ase_atoms, collect_confs
    from src.dockonsurf.clustering import clustering, get_rmsd
    from src.dockonsurf.calculation import run_calc, check_finished_calcs
    from src.dockonsurf.refinement import select_stable_confs

    logger.info('Carrying out procedures for the isolated molecule.')
    # Read the molecule
    rd_mol = adapt_format('rdkit', inp_vars['molec_file'],
                          inp_vars['special_atoms'])
    # Generate conformers
    confs = gen_confs(rd_mol, inp_vars['num_conformers'])
    # Pre-optimizes conformers
    if inp_vars['pre_opt']:
        confs, confs_ener = pre_opt_confs(confs, inp_vars['pre_opt'])
    else:
        confs_ener = pre_opt_confs(confs, max_iters=0)
    conf_list = confs_to_mol_list(confs)
    # Calculates RMSD matrix of the conformers
    rmsd_mtx = get_rmsd(conf_list)
    confs_moi = get_moments_of_inertia(confs)
    # Clusters the conformers and selects a representative
    exemplars = clustering(rmsd_mtx)
    mol_list = confs_to_mol_list(confs, exemplars)
    ase_atms_list = [rdkit_mol_to_ase_atoms(mol) for mol in mol_list]
    if len(ase_atms_list) == 0:
        err_msg = "No configurations were generated: Check the parameters in" \
                  "dockonsurf.inp"
        logger.error(err_msg)
        raise ValueError(err_msg)
    # Runs the jobs.
    run_calc('isolated', inp_vars, ase_atms_list)
    logger.info("Finished the procedures for the isolated molecule section. ")
    if inp_vars["batch_q_sys"]:
        finished_calcs, failed_calcs = check_finished_calcs('isolated',
                                                            inp_vars['code'])
        conf_list = collect_confs(finished_calcs, inp_vars['code'], 'isolated',
                                  inp_vars['special_atoms'])
        most_stable_conf = select_stable_confs(conf_list, 0)[0]

        logger.info(f"Most stable conformers is {most_stable_conf.info['iso']},"
                    f" with a total energy of {most_stable_conf.info['energy']}"
                    f" eV.")
