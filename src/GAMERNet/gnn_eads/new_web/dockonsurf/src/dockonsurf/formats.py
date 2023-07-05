"""Module for the conversion between different kinds of atomic data.

functions:
confs_to_mol_list: Converts the conformers inside a rdkit.Mol object to a list
    of separate rdkit.Mol objects.
rdkit_mol_to_ase_atoms: Converts a rdkit.Mol object into ase.Atoms object.
add_special_atoms: Allows ase to use custom elements with symbols not in the
    periodic table.
adapt_format: Converts the coordinate files into a required library object type.
read_coords_cp2k: Reads the coordinates from a CP2K restart file and returns an
    ase.Atoms object.
read_coords_vasp: Reads the coordinates from VASP OUTCAR file and returns an
    ase.Atoms object.
read_energy_cp2k: Reads the CP2K out file and returns its final energy.
collect_confs: Reads the coordinates and energies of a list of finished
    calculations.
"""

import logging

import rdkit.Chem.AllChem as Chem

logger = logging.getLogger('DockOnSurf')


def confs_to_mol_list(mol: Chem.rdchem.Mol, idx_lst=None):
    """Converts the conformers inside a rdkit mol object to a list of
    separate mol objects.

    @param mol: rdkit mol object containing at least one conformer.
    @param idx_lst: list of conformer indices to be considered. If not passed,
        all conformers are considered.
    @return: list of separate mol objects.
    """
    if idx_lst is None:
        idx_lst = list(range(mol.GetNumConformers()))
    return [Chem.MolFromMolBlock(
        Chem.MolToMolBlock(mol, confId=int(idx)).replace("3D", ""),
        removeHs=False) for idx in idx_lst]


def rdkit_mol_to_ase_atoms(mol: Chem.rdchem.Mol):
    """Converts a rdkit mol object into ase Atoms object.
    @param mol: rdkit mol object containing only one conformer.
    @return ase.Atoms: ase Atoms object with the same coordinates.
    """
    from ase import Atoms
    if mol.GetNumConformers() > 1:
        logger.warning('A mol object with multiple conformers is parsed, '
                       'converting to Atoms only the first conformer.')
    symbols = [atm.GetSymbol() for atm in mol.GetAtoms()]
    positions = mol.GetConformer(0).GetPositions()
    return Atoms(symbols=symbols, positions=positions)


def add_special_atoms(symbol_pairs):
    """Allows ase to use custom elements with symbols not in the periodic table.

    This function adds new chemical elements to be used by ase. Every new custom
    element must have a traditional (present in the periodic table) partner
    from which to obtain all its properties.
    @param symbol_pairs: List of tuples containing pairs of chemical symbols.
        Every tuple contains a pair of chemical symbols, the first label must be
        the label of the custom element and the second one the symbol of the
        reference one (traditional present on the periodic table).
    @return:
    """  # TODO Enable special atoms for rdkit
    import numpy as np
    from ase import data
    for i, pair in enumerate(symbol_pairs):
        data.chemical_symbols += [pair[0]]
        z_orig = data.atomic_numbers[pair[1]]
        orig_iupac_mass = data.atomic_masses_iupac2016[z_orig]
        orig_com_mass = data.atomic_masses_common[z_orig]
        data.atomic_numbers[pair[0]] = max(data.atomic_numbers.values()) + 1
        data.atomic_names += [pair[0]]
        data.atomic_masses_iupac2016 = np.append(data.atomic_masses_iupac2016,
                                                 orig_iupac_mass)
        data.atomic_masses = data.atomic_masses_iupac2016
        data.atomic_masses_common = np.append(data.atomic_masses_common,
                                              orig_com_mass)
        data.covalent_radii = np.append(data.covalent_radii,
                                        data.covalent_radii[z_orig])
        data.reference_states += [data.reference_states[z_orig]]
        # TODO Add vdw_radii, gsmm and aml (smaller length)


def adapt_format(requirement, coord_file, spec_atms=tuple()):
    """Converts the coordinate files into a required library object type.

    Depending on the library required to use and the file type, it converts the
    coordinate file into a library-workable object.
    @param requirement: str, the library for which the conversion should be
        made. Accepted values: 'ase', 'rdkit'.
    @param coord_file: str, path to the coordinates file aiming to convert.
        Accepted file formats: all file formats readable by ase.
    @param spec_atms: List of tuples containing pairs of new/traditional
        chemical symbols.
    @return: an object the required library can work with.
    """
    import ase.io
    from ase.io.formats import filetype

    from src.dockonsurf.utilities import try_command

    req_vals = ['rdkit', 'ase']
    lib_err = f"The conversion to the '{requirement}' library object type" \
              f" has not yet been implemented"
    conv_info = f"Converted {coord_file} to {requirement} object type"

    fil_type_err = f'The {filetype(coord_file)} file format is not supported'

    if requirement not in req_vals:
        logger.error(lib_err)
        raise NotImplementedError(lib_err)

    if requirement == 'rdkit':
        from src.dockonsurf.xyz2mol import xyz2mol
        if filetype(coord_file) == 'xyz':  # TODO Include detection of charges.
            ase_atms = ase.io.read(coord_file)
            atomic_nums = ase_atms.get_atomic_numbers().tolist()
            xyz_coordinates = ase_atms.positions.tolist()
            rd_mol_obj = xyz2mol(atomic_nums, xyz_coordinates)
            logger.debug(conv_info)
            return Chem.AddHs(rd_mol_obj)
        elif filetype(coord_file) == 'mol':
            logger.debug(conv_info)
            return Chem.AddHs(Chem.MolFromMolFile(coord_file, removeHs=False))
        else:
            ase_atms = try_command(ase.io.read,
                                   [(ase.io.formats.UnknownFileTypeError,
                                     fil_type_err)],
                                   coord_file)
            atomic_nums = ase_atms.get_atomic_numbers().tolist()
            xyz_coordinates = ase_atms.positions.tolist()
            return xyz2mol(atomic_nums, xyz_coordinates)

    if requirement == 'ase':
        add_special_atoms(spec_atms)
        if filetype(coord_file) == 'xyz':
            logger.debug(conv_info)
            return ase.io.read(coord_file)
        elif filetype(coord_file) == 'mol':
            logger.debug(conv_info)
            rd_mol = Chem.AddHs(Chem.MolFromMolFile(coord_file, removeHs=False))
            return rdkit_mol_to_ase_atoms(rd_mol)
        else:
            return try_command(ase.io.read,
                               [(ase.io.formats.UnknownFileTypeError,
                                 fil_type_err)],
                               coord_file)


def read_coords_cp2k(file, spec_atoms=tuple()):
    """Reads the coordinates from a CP2K restart file and returns an ase.Atoms
     object.

    @param file: The file to read containing the coordinates.
    @param spec_atoms: List of tuples containing the pairs of chemical symbols.
    @return: ase.Atoms object of the coordinates in the file.
    """
    import numpy as np
    from ase import Atoms
    from pycp2k import CP2K

    cp2k = CP2K()
    cp2k.parse(file)
    force_eval = cp2k.CP2K_INPUT.FORCE_EVAL_list[0]
    raw_coords = force_eval.SUBSYS.COORD.Default_keyword
    symbols = [atom.split()[0] for atom in raw_coords]
    positions = np.array([[float(coord) for coord in atom.split()[1:]]
                          for atom in raw_coords])
    if len(spec_atoms) > 0:
        add_special_atoms(spec_atoms)
    return Atoms(symbols=symbols, positions=positions)


def read_coords_vasp(file, spec_atoms=tuple()):
    """Reads the coordinates from VASP OUTCAR and returns an ase.Atoms object.

    @param file: The file to read containing the coordinates.
    @param spec_atoms: List of tuples containing the pairs of chemical symbols.
    @return: ase.Atoms object of the coordinates in the file.
    """
    import os
    import ase.io
    if not os.path.isfile(file):
        err_msg = f"File {file} not found."
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    if len(spec_atoms) > 0:
        add_special_atoms(spec_atoms)
    return ase.io.read(file)


def read_energy_cp2k(file):
    """Reads the CP2K output file and returns its final energy.

    @param file: The file from which the energy should be read.
    @return: The last energy on the out file in eV.
    """
    out_fh = open(file, 'r')
    energy = None
    for line in out_fh:
        if "ENERGY| Total FORCE_EVAL ( QS ) energy" in line:
            energy = float(line.strip().split(':')[1]) * 27.2113845  # Ha to eV
    out_fh.close()
    return energy


def collect_confs(dir_list, code, run_type, spec_atms=tuple()):
    """Reads the coordinates and energies of a list of finished calculations.

    Given a dockonsurf directory hierarchy: project/run_type/conf_X
    (run_type = ['isolated', 'screening' or 'refinement']) it reads the
    coordinates of each conf_X, it assigns its total energy from the calculation
    and assigns the conf_X label to track its origin. Finally it returns the
    ase.Atoms object.

    @param dir_list: List of directories where to read the coords from.
    @param code: the code that produced the calculation results files.
    @param run_type: the type of calculation (and also the name of the folder)
        containing the calculation subdirectories.
    @param spec_atms: List of tuples containing pairs of new/traditional
        chemical symbols.
    @return: list of ase.Atoms objects.
    """
    from glob import glob
    import os
    from src.dockonsurf.utilities import is_binary
    atoms_list = []
    for conf_dir in dir_list:
        conf_path = f"{run_type}/{conf_dir}/"
        if code == 'cp2k':
            ase_atms = read_coords_cp2k(glob(f"{conf_path}/*-1.restart")[0],
                                        spec_atms)
            # Assign energy
            for fil in os.listdir(conf_path):
                if is_binary(conf_path + fil):
                    continue
                conf_energy = read_energy_cp2k(conf_path + fil)
                if conf_energy is not None:
                    ase_atms.info["energy"] = conf_energy
                    break
            ase_atms.info[run_type[:3]] = conf_dir
            atoms_list.append(ase_atms)
        elif code == 'vasp':
            ase_atms = read_coords_vasp(f"{conf_path}/OUTCAR", spec_atms)
            ase_atms.info["energy"] = ase_atms.get_total_energy() * 27.2113845
            ase_atms.info[run_type[:3]] = conf_dir
            atoms_list.append(ase_atms)
        else:
            err_msg = f"Collect coordinates not implemented for '{code}'."
            logger.error(err_msg)
            raise NotImplementedError(err_msg)
    return atoms_list
