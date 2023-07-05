"""Functions to deal with DockOnSurf input files.

List of functions:

Auxiliary functions
-------------------
str2lst: Converts a string of integers, and groups of them, to a list of lists.

check_expect_val: Checks whether the value of an option has an adequate value.

check_inp_files: Checks if the CP2K/VASP input files are consistent.

Functions to read parameters in the Global section
--------------------------------------------------
get_run_type: Gets 'run_type' value and checks that its value is acceptable.

get_code: Gets 'code' value and checks that its value is acceptable.

get_batch_q_sys: Gets 'batch_q_sys' value and checks that its value is acceptable.

get_pbc_cell: Gets 'pbc_cell' value and checks that its value is acceptable.

get_subm_script: Gets 'subm_script' value and checks that its value is acceptable.

get_project_name: Gets 'project_name' value and checks that its value is acceptable.

get_relaunch_err: Gets 'relaunch_err' value and checks that its value is acceptable. # WARNING: OPTION NOT IMPLEMENTED

get_max_jobs: Gets 'max_jobs' value and checks that its value is acceptable.

get_special_atoms: Gets 'special_atoms' value and checks that its value is acceptable.

get_potcar_dir: Gets 'potcar_dir' value and checks that its value is acceptable.

Functions to read parameters in the Isolated section
----------------------------------------------------
get_isol_inp_file: Gets 'isol_inp_file' value and checks that its value is  acceptable.

get_molec_file: Gets 'molec_file' value and checks that its value is acceptable.

get_num_conformers: Gets 'num_conformers' value and checks that its value is  acceptable.

get_pre_opt: Gets 'pre_opt' value and checks that its value is acceptable.

Functions to read parameters in the Screening section
-----------------------------------------------------
get_screen_inp_file: Gets 'screen_inp_file' value and checks that its value is acceptable.

get_surf_file: Gets 'surf_file' value and checks that its value is acceptable.

get_sites: Gets 'sites' value and checks that its value is acceptable.

get_surf_ctrs2: Gets 'surf_ctrs2' value and checks that its value is acceptable.

get_molec_ctrs: Gets 'molec_ctrs' value and checks that its value is acceptable.

get_molec_ctrs2: Gets 'molec_ctrs2' value and checks that its value is acceptable.

get_molec_ctrs3: Gets 'molec_ctrs3' value and checks that its value is acceptable.

get_max_helic_angle: Gets 'max_helic_angle' value and checks that its value is acceptable.

get_select_magns: Gets 'select_magns' value and checks that its value is acceptable.

get_confs_per_magn: Gets 'confs_per_magn' value and checks that its value is acceptable.

get_surf_norm_vect: Gets 'surf_norm_vect' value and checks that its value is acceptable.

get_adsorption_height: Gets 'adsorption_height' value and checks that its value is acceptable.

get_set_angles: Gets 'set_angles' value and checks that its value is acceptable.

get_pts_per_angle: Gets 'pts_per_angle' value and checks that its value is acceptable.

get_max_structures: Gets 'max_structures' value and checks that its value is acceptable.

get_coll_thrsld: Gets 'coll_thrsld' value and checks that its value is acceptable.

get_min_coll_height: Gets 'coll_bottom_z' value and checks that its value is acceptable.

get_exclude_ads_ctr: Gets 'exclude_ads_ctr' value and checks that its value is acceptable.

get_H_donor: Gets 'H_donor' value and checks that its value is acceptable.

get_H_acceptor: Gets 'H_acceptor' value and checks that its value is acceptable.

get_use_molec_file: Gets 'use_molec_file' value and checks that its value is acceptable.

Functions to read parameters in the Refinement section
------------------------------------------------------
get_refine_inp_file: Gets 'refine_inp_file' value and checks that its value is acceptable.

get_energy_cutoff: Gets 'energy_cutoff' value and checks that its value is acceptable.

read_input: Directs the reading of the parameters in the input file
"""
import os.path
import logging
from configparser import ConfigParser, NoSectionError, NoOptionError, \
    MissingSectionHeaderError, DuplicateOptionError
import numpy as np
from src.dockonsurf.utilities import try_command

logger = logging.getLogger('DockOnSurf')

dos_inp = ConfigParser(inline_comment_prefixes='#',
                       empty_lines_in_values=False)
# Define new answers to be interpreted as True or False.
new_answers = {'n': False, 'none': False, 'nay': False,
               'y': True, 'sí': True, 'aye': True, 'sure': True}
for answer, val in new_answers.items():
    dos_inp.BOOLEAN_STATES[answer] = val  # TODO Check value 0
turn_false_answers = [answer for answer in dos_inp.BOOLEAN_STATES
                      if dos_inp.BOOLEAN_STATES[answer] is False]
turn_true_answers = [answer for answer in dos_inp.BOOLEAN_STATES
                     if dos_inp.BOOLEAN_STATES[answer]]

# Template error messages to be customized in place.
no_sect_err = "Section '%s' not found on input file"
no_opt_err = "Option '%s' not found on section '%s'"
num_error = "'%s' value must be a %s"


# Auxilary functions

def str2lst(cmplx_str, func=int):  # TODO: enable deeper level of nested lists
    # TODO Treat all-enclosing parenthesis as a list instead of list of lists.
    """Converts a string of integers/floats, and groups of them, to a list.

    Keyword arguments:
    @param cmplx_str: str, string of integers (or floats) and groups of them
    enclosed by parentheses-like characters.
    - Group enclosers: '()' '[]' and '{}'.
    - Separators: ',' ';' and ' '.
    - Nested groups are not allowed: '3 ((6 7) 8) 4'.
    @param func: either to use int or float

    @return list, list of integers (or floats), or list of integers (or floats)
    in the case they were grouped. First, the singlets are placed, and then the
    groups in input order.

    eg. '128,(135 138;141] 87 {45, 68}' -> [128, 87, [135, 138, 141], [45, 68]]
    """

    # Checks
    error_msg = "Function argument should be a str, sequence of " \
                "numbers separated by ',' ';' or ' '." \
                "\nThey can be grouped in parentheses-like enclosers: '()', " \
                "'[]' or {}. Nested groups are not allowed. \n" \
                "eg. 128,(135 138;141) 87 {45, 68}"
    cmplx_str = try_command(cmplx_str.replace, [(AttributeError, error_msg)],
                            ',', ' ')

    cmplx_str = cmplx_str.replace(';', ' ').replace('[', '(').replace(
        ']', ')').replace('{', '(').replace('}', ')')

    try_command(list, [(ValueError, error_msg)], map(func, cmplx_str.replace(
        ')', '').replace('(', '').split()))

    depth = 0
    for el in cmplx_str.split():
        if '(' in el:
            depth += 1
        if ')' in el:
            depth += -1
        if depth > 1 or depth < 0:
            logger.error(error_msg)
            raise ValueError(error_msg)

    init_list = cmplx_str.split()
    start_group = []
    end_group = []
    for i, element in enumerate(init_list):
        if '(' in element:
            start_group.append(i)
            init_list[i] = element.replace('(', '')
        if ')' in element:
            end_group.append(i)
            init_list[i] = element.replace(')', '')

    init_list = list(map(func, init_list))

    new_list = []
    for start_el, end_el in zip(start_group, end_group):
        new_list.append(init_list[start_el:end_el + 1])

    for v in new_list:
        for el in v:
            init_list.remove(el)
    return init_list + new_list


def check_expect_val(value, expect_vals, err_msg=None):
    """Checks whether an option lies within its expected values.

    Keyword arguments:
    @param value: The variable to check if its value lies within the expected
    ones
    @param expect_vals: list, list of values allowed for the present option.
    @param err_msg: The error message to be prompted in both log and screen.
    @raise ValueError: if the value is not among the expected ones.
    @return True if the value is among the expected ones.
    """
    if err_msg is None:
        err_msg = f"'{value}' is not an adequate value.\n" \
                  f"Adequate values: {expect_vals}"
    if not any([exp_val == value for exp_val in expect_vals]):
        logger.error(err_msg)
        raise ValueError(err_msg)

    return True


def check_inp_files(inp_files, code: str, potcar_dir=None):
    """Checks if the CP2K/VASP input files are consistent.

    @param inp_files: List of input files
    @param code: The code for which the input files are for (VASP or CP2K).
    @param potcar_dir: The path where POTCARs are found
    @return: None
    """

    if code == 'cp2k':
        from pycp2k import CP2K
        if not isinstance(inp_files, str):
            err_msg = "When using CP2K, only one input file is allowed"
            logger.error(err_msg)
            ValueError(err_msg)
        elif not os.path.isfile(inp_files):
            err_msg = f"Input file {inp_files} was not found."
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        cp2k = CP2K()
        try_command(cp2k.parse,
                    [(UnboundLocalError, "Invalid CP2K input file")], inp_files)
    elif code == "vasp":
        if not potcar_dir:
            mand_files = ["INCAR", "KPOINTS"]
        else:
            mand_files = ["INCAR", "KPOINTS"]
            if any("POTCAR" in inp_file for inp_file in inp_files):
                logger.warning("A POTCAR file was specified as input file "
                               "while the automatic generation of POTCAR was "
                               "also enabled via the 'potcar_dir' keyword. The "
                               "POTCAR specified as input_file will be used "
                               "instead of the auto-generated one.")
        # Check that if inp_files is a list of file paths
        if not isinstance(inp_files, list) and all(isinstance(inp_file, str)
                                                   for inp_file in inp_files):
            err_msg = "'inp_files' should be a list of file names/paths"
            logger.error(err_msg)
            raise ValueError(err_msg)
        # Check that all mandatory files are defined once and just once.
        elif [[mand_file in inp_file for inp_file in inp_files].count(True)
              for mand_file in mand_files].count(1) != len(mand_files):
            err_msg = f"Each of the mandatory files {mand_files} must be " \
                      f"defined once and just once."
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        # Check that the defined files exist
        elif any(not os.path.isfile(inp_file) for inp_file in inp_files):
            err_msg = f"At least one of the mandatory files {mand_files} was " \
                      "not found."
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        # Check that mandatory files are actual vasp files.
        else:
            from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar
            for inp_file in inp_files:
                file_name = inp_file.split("/")[-1]
                if not any(mand_file in file_name for mand_file in mand_files):
                    continue
                file_type = ""
                for mand_file in mand_files:
                    if mand_file in inp_file:
                        file_type = mand_file
                err = False
                err_msg = f"'{inp_file}' is not a valid {file_name} file."
                try:
                    eval(file_type.capitalize()).from_file(inp_file)
                except ValueError:
                    #logger.error(err_msg)
                    err = ValueError(err_msg)
                except IndexError:
                    #logger.error(err_msg)
                    err = IndexError(err_msg)
                else:
                    if file_name == "INCAR":
                        Incar.from_file(inp_file).check_params()
                finally:
                    if isinstance(err, BaseException):
                        raise err


# Global

def get_run_type():
    isolated, screening, refinement = (False, False, False)
    run_type_vals = ['isolated', 'screening', 'refinement', 'adsorption',
                     'full']
    run_types = dos_inp.get('Global', 'run_type').split()
    for run_type in run_types:
        check_expect_val(run_type.lower(), run_type_vals)
        if 'isol' in run_type.lower():
            isolated = True
        if 'screen' in run_type.lower():
            screening = True
        if 'refine' in run_type.lower():
            refinement = True
        if 'adsor' in run_type.lower():
            screening, refinement = (True, True)
        if 'full' in run_type.lower():
            isolated, screening, refinement = (True, True, True)

    return isolated, screening, refinement


def get_code():
    code_vals = ['cp2k', 'vasp']
    check_expect_val(dos_inp.get('Global', 'code').lower(), code_vals)
    code = dos_inp.get('Global', 'code').lower()
    return code


def get_batch_q_sys():
    batch_q_sys_vals = ['sge', 'slurm', 'lsf', 'irene', 'local'] + turn_false_answers
    check_expect_val(dos_inp.get('Global', 'batch_q_sys').lower(),
                     batch_q_sys_vals)
    batch_q_sys = dos_inp.get('Global', 'batch_q_sys').lower()
    if batch_q_sys.lower() in turn_false_answers:
        return False
    else:
        return batch_q_sys


def get_pbc_cell():
    from ase.atoms import Cell
    err_msg = "'pbc_cell' must be either 3 vectors of size 3 or False."
    pbc_cell_str = dos_inp.get('Global', 'pbc_cell', fallback="False")
    if pbc_cell_str.lower() in turn_false_answers:
        return False
    else:
        pbc_cell = np.array(try_command(str2lst, [(ValueError, err_msg)],
                                        pbc_cell_str, float))
        if pbc_cell.shape != (3, 3):
            logger.error(err_msg)
            raise ValueError(err_msg)
        if np.linalg.det(pbc_cell) == 0.0:
            err_msg = "The volume of the defined cell is 0"
            logger.error(err_msg)
            raise ValueError(err_msg)
        return Cell(pbc_cell)


def get_subm_script():
    subm_script = dos_inp.get('Global', 'subm_script')
    if not os.path.isfile(subm_script):
        logger.error(f'File {subm_script} not found.')
        raise FileNotFoundError(f'File {subm_script} not found')
    return subm_script


def get_project_name():
    project_name = dos_inp.get('Global', 'project_name', fallback='')
    return project_name


def get_relaunch_err():
    # WARNING: OPTION NOT IMPLEMENTED
    relaunch_err_vals = ['geo_not_conv']
    relaunch_err = dos_inp.get('Global', 'relaunch_err',
                               fallback="False")
    if relaunch_err.lower() in turn_false_answers:
        return False
    else:
        check_expect_val(relaunch_err.lower(), relaunch_err_vals)
    return relaunch_err


def get_max_jobs():
    import re
    err_msg = "'max_jobs' must be a list of, number plus 'p', 'q' or 'r', or " \
              "a combination of them without repeating letters.\n" \
              "eg: '2r 3p 4pr', '5q' or '3r 3p'"
    max_jobs_str = dos_inp.get('Global', 'max_jobs', fallback="inf").lower()
    str_vals = ["r", "p", "q", "rp", "rq", "pr", "qr"]
    max_jobs = {"r": np.inf, "p": np.inf, "rp": np.inf}
    if "inf" == max_jobs_str:
        return {"r": np.inf, "p": np.inf, "rp": np.inf}
    # Iterate over the number of requirements:
    for req in max_jobs_str.split():
        # Split numbers from letters into a list
        req_parts = re.findall(r'[a-z]+|\d+', req)
        if len(req_parts) != 2:
            logger.error(err_msg)
            raise ValueError(err_msg)
        if req_parts[0].isdecimal():
            req_parts[1] = req_parts[1].replace('q', 'p').replace('pr', 'rp')
            if req_parts[1] in str_vals and max_jobs[req_parts[1]] == np.inf:
                max_jobs[req_parts[1]] = int(req_parts[0])
        elif req_parts[1].isdecimal():
            req_parts[0] = req_parts[0].replace('q', 'p').replace('pr', 'rp')
            if req_parts[0] in str_vals and max_jobs[req_parts[0]] == np.inf:
                max_jobs[req_parts[0]] = int(req_parts[1])
        else:
            logger.error(err_msg)
            raise ValueError(err_msg)

    return max_jobs


def get_special_atoms():
    from ase.data import chemical_symbols

    spec_at_err = '\'special_atoms\' does not have an adequate format.\n' \
                  'Adequate format: (Fe1 Fe) (O1 O)'
    special_atoms = dos_inp.get('Global', 'special_atoms', fallback="False")
    if special_atoms.lower() in turn_false_answers:
        special_atoms = []
    else:
        # Converts the string into a list of tuples
        lst_tple = [tuple(pair.replace("(", "").split()) for pair in
                    special_atoms.split(")")[:-1]]
        if len(lst_tple) == 0:
            logger.error(spec_at_err)
            raise ValueError(spec_at_err)
        for i, tup in enumerate(lst_tple):
            if not isinstance(tup, tuple) or len(tup) != 2:
                logger.error(spec_at_err)
                raise ValueError(spec_at_err)
            if tup[1].capitalize() not in chemical_symbols:
                elem_err = "The second element of the couple should be an " \
                           "actual element of the periodic table"
                logger.error(elem_err)
                raise ValueError(elem_err)
            if tup[0].capitalize() in chemical_symbols:
                elem_err = "The first element of the couple is already an " \
                           "actual element of the periodic table, "
                logger.error(elem_err)
                raise ValueError(elem_err)
            for j, tup2 in enumerate(lst_tple):
                if j <= i:
                    continue
                if tup2[0] == tup[0]:
                    label_err = f'You have specified the label {tup[0]} to ' \
                                f'more than one special atom'
                    logger.error(label_err)
                    raise ValueError(label_err)
        special_atoms = lst_tple
    return special_atoms


def get_potcar_dir():
    potcar_dir = dos_inp.get('Global', 'potcar_dir', fallback="False")
    if potcar_dir.lower() in turn_false_answers:
        return False
    elif not os.path.isdir(potcar_dir):
        err_msg = "'potcar_dir' must be either False or a directory."
        logger.error(err_msg)
        raise ValueError(err_msg)
    else:
        return potcar_dir


# Isolated

def get_isol_inp_file(code, potcar_dir=None):  # TODO allow spaces in path names
    inp_file_lst = dos_inp.get('Isolated', 'isol_inp_file').split()
    check_inp_files(inp_file_lst[0] if len(inp_file_lst) == 1 else inp_file_lst,
                    code, potcar_dir)
    return inp_file_lst[0] if len(inp_file_lst) == 1 else inp_file_lst


def get_molec_file():
    molec_file = dos_inp.get('Isolated', 'molec_file')
    if not os.path.isfile(molec_file):
        logger.error(f'File {molec_file} not found.')
        raise FileNotFoundError(f'File {molec_file} not found')
    return molec_file


def get_num_conformers():
    err_msg = num_error % ('num_conformers', 'positive integer')
    num_conformers = try_command(dos_inp.getint, [(ValueError, err_msg)],
                                 'Isolated', 'num_conformers', fallback=100)
    if num_conformers < 1:
        logger.error(err_msg)
        raise ValueError(err_msg)
    return num_conformers


def get_pre_opt():
    pre_opt_vals = ['uff', 'mmff'] + turn_false_answers
    check_expect_val(dos_inp.get('Isolated', 'pre_opt').lower(), pre_opt_vals)
    pre_opt = dos_inp.get('Isolated', 'pre_opt').lower()
    if pre_opt in turn_false_answers:
        return False
    else:
        return pre_opt


# Screening

def get_screen_inp_file(code,
                        potcar_dir=None):  # TODO allow spaces in path names
    inp_file_lst = dos_inp.get('Screening', 'screen_inp_file').split()
    check_inp_files(inp_file_lst[0] if len(inp_file_lst) == 1 else inp_file_lst,
                    code, potcar_dir)
    return inp_file_lst[0] if len(inp_file_lst) == 1 else inp_file_lst


def get_surf_file():
    surf_file = dos_inp.get('Screening', 'surf_file')
    if not os.path.isfile(surf_file):
        logger.error(f'File {surf_file} not found.')
        raise FileNotFoundError(f'File {surf_file} not found')
    return surf_file


def get_sites():
    err_msg = 'The value of sites should be a list of atom numbers ' \
              '(ie. positive integers) or groups of atom numbers ' \
              'grouped by parentheses-like enclosers. \n' \
              'eg. 128,(135 138;141) 87 {45, 68}'
    # Convert the string into a list of lists
    sites = try_command(str2lst,
                        [(ValueError, err_msg), (AttributeError, err_msg)],
                        dos_inp.get('Screening', 'sites'))
    # Check all elements of the list (of lists) are positive integers
    for site in sites:
        if type(site) is list:
            for atom in site:
                if atom < 0:
                    logger.error(err_msg)
                    raise ValueError(err_msg)
        elif type(site) is int:
            if site < 0:
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            logger.error(err_msg)
            raise ValueError(err_msg)

    return sites


def get_surf_ctrs2():
    err_msg = 'The value of surf_ctrs2 should be a list of atom numbers ' \
              '(ie. positive integers) or groups of atom numbers ' \
              'grouped by parentheses-like enclosers. \n' \
              'eg. 128,(135 138;141) 87 {45, 68}'
    # Convert the string into a list of lists
    surf_ctrs2 = try_command(str2lst,
                             [(ValueError, err_msg), (AttributeError, err_msg)],
                             dos_inp.get('Screening', 'surf_ctrs2'))
    # Check all elements of the list (of lists) are positive integers
    for ctr in surf_ctrs2:
        if type(ctr) is list:
            for atom in ctr:
                if atom < 0:
                    logger.error(err_msg)
                    raise ValueError(err_msg)
        elif type(ctr) is int:
            if ctr < 0:
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            logger.error(err_msg)
            raise ValueError(err_msg)

    return surf_ctrs2


def get_molec_ctrs():
    err_msg = 'The value of molec_ctrs should be a list of atom' \
              ' numbers (ie. positive integers) or groups of atom ' \
              'numbers enclosed by parentheses-like characters. \n' \
              'eg. 128,(135 138;141) 87 {45, 68}'
    # Convert the string into a list of lists
    molec_ctrs = try_command(str2lst,
                             [(ValueError, err_msg),
                              (AttributeError, err_msg)],
                             dos_inp.get('Screening', 'molec_ctrs'))
    # Check all elements of the list (of lists) are positive integers
    for ctr in molec_ctrs:
        if isinstance(ctr, list):
            for atom in ctr:
                if atom < 0:
                    logger.error(err_msg)
                    raise ValueError(err_msg)
        elif isinstance(ctr, int):
            if ctr < 0:
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            logger.error(err_msg)
            raise ValueError(err_msg)

    return molec_ctrs


def get_molec_ctrs2():
    err_msg = 'The value of molec_ctrs2 should be a list of atom ' \
              'numbers (ie. positive integers) or groups of atom ' \
              'numbers enclosed by parentheses-like characters. \n' \
              'eg. 128,(135 138;141) 87 {45, 68}'
    # Convert the string into a list of lists
    molec_ctrs2 = try_command(str2lst, [(ValueError, err_msg),
                                        (AttributeError, err_msg)],
                              dos_inp.get('Screening', 'molec_ctrs2'))

    # Check all elements of the list (of lists) are positive integers
    for ctr in molec_ctrs2:
        if isinstance(ctr, list):
            for atom in ctr:
                if atom < 0:
                    logger.error(err_msg)
                    raise ValueError(err_msg)
        elif isinstance(ctr, int):
            if ctr < 0:
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            logger.error(err_msg)
            raise ValueError(err_msg)

    return molec_ctrs2


def get_molec_ctrs3():
    err_msg = 'The value of molec_ctrs3 should be a list of atom ' \
              'numbers (ie. positive integers) or groups of atom ' \
              'numbers enclosed by parentheses-like characters. \n' \
              'eg. 128,(135 138;141) 87 {45, 68}'
    # Convert the string into a list of lists
    molec_ctrs3 = try_command(str2lst, [(ValueError, err_msg),
                                        (AttributeError, err_msg)],
                              dos_inp.get('Screening', 'molec_ctrs3'))

    # Check all elements of the list (of lists) are positive integers
    for ctr in molec_ctrs3:
        if isinstance(ctr, list):
            for atom in ctr:
                if atom < 0:
                    logger.error(err_msg)
                    raise ValueError(err_msg)
        elif isinstance(ctr, int):
            if ctr < 0:
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            logger.error(err_msg)
            raise ValueError(err_msg)

    return molec_ctrs3


def get_max_helic_angle():
    err_msg = "'max_helic_angle' must be a positive number in degrees"
    max_helic_angle = try_command(dos_inp.getfloat, [(ValueError, err_msg)],
                                  'Screening', 'max_helic_angle',
                                  fallback=180.0)
    if max_helic_angle < 0:
        logger.error(err_msg)
        raise ValueError(err_msg)

    return max_helic_angle


def get_select_magns():
    select_magns_vals = ['energy', 'moi']
    select_magns_str = dos_inp.get('Screening', 'select_magns',
                                   fallback='moi')
    select_magns_str.replace(',', ' ').replace(';', ' ')
    select_magns = select_magns_str.split(' ')
    select_magns = [m.lower() for m in select_magns]
    for m in select_magns:
        check_expect_val(m, select_magns_vals)
    return select_magns


def get_confs_per_magn():
    err_msg = num_error % ('confs_per_magn', 'positive integer')
    confs_per_magn = try_command(dos_inp.getint, [(ValueError, err_msg)],
                                 'Screening', 'confs_per_magn', fallback=2)
    if confs_per_magn <= 0:
        logger.error(err_msg)
        raise ValueError(err_msg)
    return confs_per_magn


def get_surf_norm_vect():
    err = "'surf_norm_vect' must be a 3 component vector, 'x', 'y' or 'z', " \
          "'auto' or 'asann'."
    cart_axes = {'x': [1.0, 0.0, 0.0], '-x': [-1.0, 0.0, 0.0],
                 'y': [0.0, 1.0, 0.0], '-y': [0.0, -1.0, 0.0],
                 'z': [0.0, 0.0, 1.0], '-z': [0.0, 0.0, -1.0]}
    surf_norm_vect_str = dos_inp.get('Screening', 'surf_norm_vect',
                                     fallback="auto").lower()
    if surf_norm_vect_str == "asann" or surf_norm_vect_str == "auto":
        return 'auto'
    if surf_norm_vect_str in cart_axes:
        return np.array(cart_axes[surf_norm_vect_str])
    surf_norm_vect = try_command(str2lst, [(ValueError, err)],
                                 surf_norm_vect_str, float)
    if len(surf_norm_vect) != 3:
        logger.error(err)
        raise ValueError(err)

    return np.array(surf_norm_vect) / np.linalg.norm(surf_norm_vect)


def get_adsorption_height():
    err_msg = num_error % ('adsorption_height', 'positive number')
    ads_height = try_command(dos_inp.getfloat, [(ValueError, err_msg)],
                             'Screening', 'adsorption_height', fallback=2.5)
    if ads_height <= 0:
        logger.error(err_msg)
        raise ValueError(err_msg)
    return ads_height


def get_set_angles():
    set_vals = ['euler', 'internal']
    check_expect_val(dos_inp.get('Screening', 'set_angles').lower(), set_vals)
    set_angles = dos_inp.get('Screening', 'set_angles',
                             fallback='euler').lower()
    return set_angles


def get_pts_per_angle():
    err_msg = num_error % ('sample_points_per_angle', 'positive integer')
    pts_per_angle = try_command(dos_inp.getint,
                                [(ValueError, err_msg)],
                                'Screening', 'sample_points_per_angle',
                                fallback=3)
    if pts_per_angle <= 0:
        logger.error(err_msg)
        raise ValueError(err_msg)
    return pts_per_angle


def get_max_structures():
    err_msg = num_error % ('max_structures', 'positive integer')
    max_structs = dos_inp.get('Screening', 'max_structures', fallback="False")
    if max_structs.lower() in turn_false_answers:
        return np.inf
    if try_command(int, [(ValueError, err_msg)], max_structs) <= 0:
        logger.error(err_msg)
        raise ValueError(err_msg)
    return int(max_structs)


def get_coll_thrsld():
    err_msg = num_error % ('collision_threshold', 'positive number')
    coll_thrsld_str = dos_inp.get('Screening', 'collision_threshold',
                                  fallback="False")
    if coll_thrsld_str.lower() in turn_false_answers:
        return False
    coll_thrsld = try_command(float, [(ValueError, err_msg)], coll_thrsld_str)

    if coll_thrsld <= 0:
        logger.error(err_msg)
        raise ValueError(err_msg)

    return coll_thrsld


def get_min_coll_height(norm_vect):
    err_msg = num_error % ('min_coll_height', 'decimal number')
    min_coll_height = dos_inp.get('Screening', 'min_coll_height',
                                  fallback="false")
    if min_coll_height.lower() in turn_false_answers:
        return False
    min_coll_height = try_command(float, [(ValueError, err_msg)],
                                  min_coll_height)
    cart_axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                 [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    err_msg = "'min_coll_height' option is only implemented for " \
              "'surf_norm_vect' to be one of the x, y or z axes. "
    if not isinstance(norm_vect, str) or norm_vect != 'auto':
        check_expect_val(norm_vect.tolist(), cart_axes, err_msg)
    return min_coll_height


def get_exclude_ads_ctr():
    err_msg = "exclude_ads_ctr must have a boolean value."
    exclude_ads_ctr = try_command(dos_inp.getboolean, [(ValueError, err_msg)],
                                  "Screening", "exclude_ads_ctr",
                                  fallback=False)
    return exclude_ads_ctr


def get_H_donor(spec_atoms):
    from ase.data import chemical_symbols
    err_msg = "The value of 'h_donor' must be either False, a chemical symbol " \
              "or an atom index"
    h_donor_str = dos_inp.get('Screening', 'h_donor', fallback="False")
    h_donor = []
    if h_donor_str.lower() in turn_false_answers:
        return False
    err = False
    for el in h_donor_str.split():
        try:
            h_donor.append(int(el))
        except ValueError:
            if el not in chemical_symbols + [nw_sym for pairs in spec_atoms
                                             for nw_sym in pairs]:
                err = True
            else:
                h_donor.append(el)
        finally:
            if err:
                logger.error(err_msg)
                ValueError(err_msg)
    return h_donor


def get_H_acceptor(spec_atoms):
    from ase.data import chemical_symbols
    err_msg = "The value of 'h_acceptor' must be either 'all', a chemical " \
              "symbol or an atom index."
    h_acceptor_str = dos_inp.get('Screening', 'h_acceptor', fallback="all")
    if h_acceptor_str.lower() == "all":
        return "all"
    h_acceptor = []
    err = False
    for el in h_acceptor_str.split():
        try:
            h_acceptor.append(int(el))
        except ValueError:
            if el not in chemical_symbols + [nw_sym for pairs in spec_atoms
                                             for nw_sym in pairs]:
                err = True
            else:
                h_acceptor.append(el)
        finally:
            if err:
                logger.error(err_msg)
                raise ValueError(err_msg)
    return h_acceptor


def get_use_molec_file():
    use_molec_file = dos_inp.get('Screening', 'use_molec_file',
                                 fallback='False')
    if use_molec_file.lower() in turn_false_answers:
        return False
    if not os.path.isfile(use_molec_file):
        logger.error(f'File {use_molec_file} not found.')
        raise FileNotFoundError(f'File {use_molec_file} not found')

    return use_molec_file


# Refinement

def get_refine_inp_file(code, potcar_dir=None):
    inp_file_lst = dos_inp.get('Refinement', 'refine_inp_file').split()
    check_inp_files(inp_file_lst[0] if len(inp_file_lst) == 1 else inp_file_lst,
                    code, potcar_dir)
    return inp_file_lst[0] if len(inp_file_lst) == 1 else inp_file_lst


def get_energy_cutoff():
    err_msg = num_error % ('energy_cutoff', 'positive decimal number')
    energy_cutoff = try_command(dos_inp.getfloat,
                                [(ValueError, err_msg)],
                                'Refinement', 'energy_cutoff', fallback=0.5)
    if energy_cutoff < 0:
        logger.error(err_msg)
        raise ValueError(err_msg)
    return energy_cutoff


# Read input parameters

def read_input(in_file):
    """Directs the reading of the parameters in the input file.

    @param in_file: The path to the DockOnSurf input file.
    @return inp_vars: Dictionary with the values for every option in the input
    file.
    """
    from src.dockonsurf.formats import adapt_format

    # Checks for errors in the Input file.
    err_msg = False
    try:
        dos_inp.read(in_file)
    except MissingSectionHeaderError as e:
        logger.error('There are options in the input file without a Section '
                     'header.')
        err_msg = e
    except DuplicateOptionError as e:
        logger.error('There is an option in the input file that has been '
                     'specified more than once.')
        err_msg = e
    except Exception as e:
        err_msg = e
    else:
        err_msg = False
    finally:
        if isinstance(err_msg, BaseException):
            raise err_msg

    inp_vars = {}

    # Global
    if not dos_inp.has_section('Global'):
        logger.error(no_sect_err % 'Global')
        raise NoSectionError('Global')

    # Mandatory options
    # Checks whether the mandatory options 'run_type', 'code', etc. are present.
    glob_mand_opts = ['run_type', 'code', 'batch_q_sys']
    for opt in glob_mand_opts:
        if not dos_inp.has_option('Global', opt):
            logger.error(no_opt_err % (opt, 'Global'))
            raise NoOptionError(opt, 'Global')

    # Mandatory options
    isolated, screening, refinement = get_run_type()
    inp_vars['isolated'] = isolated
    inp_vars['screening'] = screening
    inp_vars['refinement'] = refinement
    inp_vars['code'] = get_code()
    inp_vars['batch_q_sys'] = get_batch_q_sys()

    # Dependent options:
    if inp_vars['batch_q_sys']:
        inp_vars['max_jobs'] = get_max_jobs()
        if inp_vars['batch_q_sys'] != 'local':
            if not dos_inp.has_option('Global', 'subm_script'):
                logger.error(no_opt_err % ('subm_script', 'Global'))
                raise NoOptionError('subm_script', 'Global')
            inp_vars['subm_script'] = get_subm_script()
    if inp_vars['code'] == "vasp":
        inp_vars['potcar_dir'] = get_potcar_dir()

    # Facultative options (Default/Fallback value present)
    inp_vars['pbc_cell'] = get_pbc_cell()
    inp_vars['project_name'] = get_project_name()
    # inp_vars['relaunch_err'] = get_relaunch_err()
    inp_vars['special_atoms'] = get_special_atoms()

    # Isolated
    if isolated:
        if not dos_inp.has_section('Isolated'):
            logger.error(no_sect_err % 'Isolated')
            raise NoSectionError('Isolated')
        # Mandatory options
        iso_mand_opts = ['isol_inp_file', 'molec_file']
        for opt in iso_mand_opts:
            if not dos_inp.has_option('Isolated', opt):
                logger.error(no_opt_err % (opt, 'Isolated'))
                raise NoOptionError(opt, 'Isolated')
        if 'potcar_dir' in inp_vars:
            inp_vars['isol_inp_file'] = get_isol_inp_file(inp_vars['code'],
                                                          inp_vars[
                                                              'potcar_dir'])
        else:
            inp_vars['isol_inp_file'] = get_isol_inp_file(inp_vars['code'])
        inp_vars['molec_file'] = get_molec_file()

        # Checks for PBC
        atms = adapt_format('ase', inp_vars['molec_file'],
                            inp_vars['special_atoms'])
        if inp_vars['code'] == 'vasp' and np.linalg.det(atms.cell) == 0.0 \
                and inp_vars['pbc_cell'] is False:
            err_msg = "When running calculations with 'VASP', the PBC cell " \
                      "should be provided either implicitely inside " \
                      "'molec_file' or by setting the 'pbc_cell' option."
            logger.error(err_msg)
            raise ValueError(err_msg)
        elif inp_vars['pbc_cell'] is False and np.linalg.det(atms.cell) != 0.0:
            inp_vars['pbc_cell'] = atms.cell
            logger.info(f"Obtained pbc_cell from '{inp_vars['molec_file']}' "
                        f"file.")
        elif (atms.cell != 0).any() and not np.allclose(inp_vars['pbc_cell'],
                                                        atms.cell):
            logger.warning("'molec_file' has an implicit cell defined "
                           f"different than 'pbc_cell'.\n"
                           f"'molec_file' = {atms.cell}.\n"
                           f"'pbc_cell' = {inp_vars['pbc_cell']}).\n"
                           "'pbc_cell' value will be used.")

        # Facultative options (Default/Fallback value present)
        inp_vars['num_conformers'] = get_num_conformers()
        inp_vars['pre_opt'] = get_pre_opt()

    # Screening
    if screening:
        if not dos_inp.has_section('Screening'):
            logger.error(no_sect_err % 'Screening')
            raise NoSectionError('Screening')
        # Mandatory options:
        # Checks whether the mandatory options are present.
        # Mandatory options
        screen_mand_opts = ['screen_inp_file', 'surf_file', 'sites',
                            'molec_ctrs']
        for opt in screen_mand_opts:
            if not dos_inp.has_option('Screening', opt):
                logger.error(no_opt_err % (opt, 'Screening'))
                raise NoOptionError(opt, 'Screening')
        if 'potcar_dir' in inp_vars:
            inp_vars['screen_inp_file'] = get_screen_inp_file(inp_vars['code'],
                                                              inp_vars[
                                                                  'potcar_dir'])
        else:
            inp_vars['screen_inp_file'] = get_screen_inp_file(inp_vars['code'])
        inp_vars['surf_file'] = get_surf_file()
        inp_vars['sites'] = get_sites()
        inp_vars['molec_ctrs'] = get_molec_ctrs()

        # Checks for PBC
        atms = adapt_format('ase', inp_vars['surf_file'],
                            inp_vars['special_atoms'])
        if inp_vars['code'] == 'vasp' and np.linalg.det(atms.cell) == 0.0 \
                and inp_vars['pbc_cell'] is False:
            err_msg = "When running calculations with 'VASP', the PBC cell " \
                      "should be provided either implicitely inside " \
                      "'surf_file' or by setting the 'pbc_cell' option."
            logger.error(err_msg)
            raise ValueError(err_msg)
        elif inp_vars['pbc_cell'] is False and np.linalg.det(atms.cell) != 0.0:
            inp_vars['pbc_cell'] = atms.cell
            logger.info(f"Obtained pbc_cell from '{inp_vars['surf_file']}' "
                        f"file.")
        elif np.linalg.det(atms.cell) != 0 \
                and not np.allclose(inp_vars['pbc_cell'], atms.cell):
            logger.warning("'surf_file' has an implicit cell defined "
                           f"different than 'pbc_cell'.\n"
                           f"'surf_file' = {atms.cell}.\n"
                           f"'pbc_cell' = {inp_vars['pbc_cell']}).\n"
                           "'pbc_cell' value will be used.")

        # Facultative options (Default value present)
        inp_vars['select_magns'] = get_select_magns()
        inp_vars['confs_per_magn'] = get_confs_per_magn()
        inp_vars['surf_norm_vect'] = get_surf_norm_vect()
        inp_vars['adsorption_height'] = get_adsorption_height()
        inp_vars['set_angles'] = get_set_angles()
        inp_vars['sample_points_per_angle'] = get_pts_per_angle()
        inp_vars['collision_threshold'] = get_coll_thrsld()
        inp_vars['min_coll_height'] = get_min_coll_height(
            inp_vars['surf_norm_vect'])
        if inp_vars['min_coll_height'] is False \
                and inp_vars['collision_threshold'] is False:
            logger.warning("Collisions are deactivated: Overlapping of "
                           "adsorbate and surface is possible")
        inp_vars['exclude_ads_ctr'] = get_exclude_ads_ctr()
        inp_vars['h_donor'] = get_H_donor(inp_vars['special_atoms'])
        inp_vars['max_structures'] = get_max_structures()
        inp_vars['use_molec_file'] = get_use_molec_file()

        # Options depending on the value of others
        if inp_vars['set_angles'] == "internal":
            internal_opts = ['molec_ctrs2', 'molec_ctrs3', 'surf_ctrs2',
                             'max_helic_angle']
            for opt in internal_opts:
                if not dos_inp.has_option('Screening', opt):
                    logger.error(no_opt_err % (opt, 'Screening'))
                    raise NoOptionError(opt, 'Screening')
            inp_vars['max_helic_angle'] = get_max_helic_angle()
            inp_vars['molec_ctrs2'] = get_molec_ctrs2()
            inp_vars['molec_ctrs3'] = get_molec_ctrs3()
            inp_vars['surf_ctrs2'] = get_surf_ctrs2()
            if len(inp_vars["molec_ctrs2"]) != len(inp_vars['molec_ctrs']) \
                    or len(inp_vars["molec_ctrs3"]) != \
                    len(inp_vars['molec_ctrs']) \
                    or len(inp_vars['surf_ctrs2']) != len(inp_vars['sites']):
                err_msg = "'molec_ctrs' 'molec_ctrs2' and 'molec_ctrs3' must " \
                          "have the same number of indices "
                logger.error(err_msg)
                raise ValueError(err_msg)

        if inp_vars['h_donor'] is False:
            inp_vars['h_acceptor'] = False
        else:
            inp_vars['h_acceptor'] = get_H_acceptor(inp_vars['special_atoms'])

    # Refinement
    if refinement:
        if not dos_inp.has_section('Refinement'):
            logger.error(no_sect_err % 'Refinement')
            raise NoSectionError('Refinement')
        # Mandatory options
        # Checks whether the mandatory options are present.
        ref_mand_opts = ['refine_inp_file']
        for opt in ref_mand_opts:
            if not dos_inp.has_option('Refinement', opt):
                logger.error(no_opt_err % (opt, 'Refinement'))
                raise NoOptionError(opt, 'Refinement')
        if 'potcar_dir' in inp_vars:
            inp_vars['refine_inp_file'] = get_refine_inp_file(inp_vars['code'],
                                                              inp_vars[
                                                                  'potcar_dir'])
        else:
            inp_vars['refine_inp_file'] = get_refine_inp_file(inp_vars['code'])

        # Facultative options (Default value present)
        inp_vars['energy_cutoff'] = get_energy_cutoff()
        # end energy_cutoff

    return_vars_str = "\n\t".join([str(key) + ": " + str(value)
                                   for key, value in inp_vars.items()])
    logger.info(f'Correctly read {in_file} parameters:'
                f' \n\n\t{return_vars_str}\n')

    return inp_vars


if __name__ == "__main__":
    import sys

    print(read_input(sys.argv[1]))
