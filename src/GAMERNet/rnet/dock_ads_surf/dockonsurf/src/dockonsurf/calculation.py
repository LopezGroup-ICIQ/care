"""

*Functions*

check_finished_calcs: Checks if the calculations finished normally or not.

prep_cp2k: Prepares the directories to run calculations with CP2K.

prep_vasp: Prepares the directories to run calculations with VASP.

get_jobs_status: Returns a list of job status for a list of job ids.

submit_jobs: Submits jobs to a custom queuing system with the provided script

run_calc: Directs calculation run/submission
"""

import os
import logging

logger = logging.getLogger('DockOnSurf')


def check_finished_calcs(run_type, code):
    """Checks if the calculations finished normally or not.

    @param run_type: The type of calculation to check.
    @param code: The code used for the specified job.
    @return finished_calcs: List of calculation directories that have finished
    normally.
    @return unfinished_calcs: List of calculation directories that have finished
    abnormally.
    """
    from glob import glob
    import ase.io
    from src.dockonsurf.utilities import tail, is_binary
    from src.dockonsurf.utilities import _human_key

    finished_calcs = []
    unfinished_calcs = []
    for conf_dir in sorted(os.listdir(run_type), key=_human_key):
        conf_path = f'{run_type}/{conf_dir}/'
        if not os.path.isdir(conf_path) or 'conf_' not in conf_dir:
            continue
        if code == 'cp2k':
            restart_file_list = glob(f"{conf_path}/*-1.restart")
            if len(restart_file_list) == 0:
                logger.warning(f"No *-1.restart file found on {conf_path}.")
                unfinished_calcs.append(conf_dir)
                continue
            elif len(restart_file_list) > 1:
                warn_msg = f'There is more than one CP2K restart file ' \
                           f'(*-1.restart / in {conf_path}: ' \
                           f'{restart_file_list}. Skipping directory.'
                unfinished_calcs.append(conf_dir)
                logger.warning(warn_msg)
                continue
            elif os.stat(restart_file_list[0]).st_size == 0:
                unfinished_calcs.append(conf_dir)
                logger.warning(f'{restart_file_list[0]} is an empty file.')
                continue
            out_files = []
            for file in os.listdir(conf_path):
                if is_binary(conf_path+file):
                    continue
                with open(conf_path+file, "rb") as out_fh:
                    tail_out_str = tail(out_fh)
                if tail_out_str.count("PROGRAM STOPPED IN") == 1:
                    out_files.append(file)
            if len(out_files) > 1:
                warn_msg = f'There is more than one CP2K output file in ' \
                           f'{conf_path}: {out_files}. Skipping directory.'
                logger.warning(warn_msg)
                unfinished_calcs.append(conf_dir)
            elif len(out_files) == 0:
                warn_msg = f'There is no CP2K output file in {conf_path}. ' \
                           'Skipping directory.'
                logger.warning(warn_msg)
                unfinished_calcs.append(conf_dir)
            else:
                finished_calcs.append(conf_dir)
        elif code == 'vasp':
            out_file_list = glob(f"{conf_path}/OUTCAR")
            if len(out_file_list) == 0:
                unfinished_calcs.append(conf_dir)
            elif len(out_file_list) > 1:
                warn_msg = f'There is more than one file matching the {code} ' \
                           f'pattern for finished calculation (*.out / ' \
                           f'*-1.restart) in {conf_path}: ' \
                           f'{out_file_list}. Skipping directory.'
                logger.warning(warn_msg)
                unfinished_calcs.append(conf_dir)
            else:
                try:
                    ase.io.read(f"{conf_path}/OUTCAR")
                except ValueError:
                    unfinished_calcs.append(conf_dir)
                    continue
                except IndexError:
                    unfinished_calcs.append(conf_dir)
                    continue
                with open(f"{conf_path}/OUTCAR", 'rb') as out_fh:
                    if "General timing and accounting" not in tail(out_fh):
                        unfinished_calcs.append(conf_dir)
                    else:
                        finished_calcs.append(conf_dir)
        else:
            err_msg = f"Check not implemented for '{code}'."
            logger.error(err_msg)
            raise NotImplementedError(err_msg)
    return finished_calcs, unfinished_calcs


def prep_cp2k(inp_file: str, run_type: str, atms_list: list, proj_name: str):
    """Prepares the directories to run calculations with CP2K.

    @param inp_file: CP2K Input file to run the calculations with.
    @param run_type: Type of calculation. 'isolated', 'screening' or
        'refinement'
    @param atms_list: list of ase.Atoms objects to run the calculation of.
    @param proj_name: name of the project
    @return: None
    """
    from shutil import copy
    from pycp2k import CP2K
    from src.dockonsurf.utilities import check_bak
    if not isinstance(inp_file, str):
        err_msg = "'inp_file' must be a string with the path of the CP2K " \
                  "input file."
        logger.error(err_msg)
        raise ValueError(err_msg)
    cp2k = CP2K()
    cp2k.parse(inp_file)
    cp2k.CP2K_INPUT.GLOBAL.Project_name = proj_name+"_"+run_type
    force_eval = cp2k.CP2K_INPUT.FORCE_EVAL_list[0]
    if force_eval.SUBSYS.TOPOLOGY.Coord_file_name is None:
        logger.warning("'COORD_FILE_NAME' not specified on CP2K input. Using\n"
                       "'coord.xyz'. A new CP2K input file with "
                       "the 'COORD_FILE_NAME' variable is created.")
        force_eval.SUBSYS.TOPOLOGY.Coord_file_name = 'coord.xyz'
        check_bak(inp_file.split('/')[-1])
    new_inp_file = inp_file.split('/')[-1]
    cp2k.write_input_file(new_inp_file)

    coord_file = force_eval.SUBSYS.TOPOLOGY.Coord_file_name

    # Creating and setting up directories for every configuration.
    for i, conf in enumerate(atms_list):
        subdir = f'{run_type}/conf_{i}/'
        os.mkdir(subdir)
        copy(new_inp_file, subdir)
        conf.write(subdir + coord_file)


def prep_vasp(inp_files, run_type, atms_list, proj_name, cell, potcar_dir):
    """Prepares the directories to run calculations with VASP.

    @param inp_files: VASP Input files to run the calculations with.
    @param run_type: Type of calculation. 'isolated', 'screening' or
        'refinement'
    @param atms_list: list of ase.Atoms objects to run the calculation of.
    @param proj_name: name of the project.
    @param cell: Cell for the Periodic Boundary Conditions.
    @param potcar_dir: Directory to find POTCARs for each element.
    @return: None
    """
    from shutil import copy
    import os

    import numpy as np
    from pymatgen.io.vasp.inputs import Incar

    if not potcar_dir:
        mand_files = ["INCAR", "KPOINTS"]
    elif any("POTCAR" in inp_file for inp_file in inp_files):
        mand_files = ["INCAR", "KPOINTS"]
    else:
        mand_files = ["INCAR", "KPOINTS"]

    # Check that there are many specified files
    if not isinstance(inp_files, list) and all(isinstance(inp_file, str)
                                               for inp_file in inp_files):
        err_msg = "'inp_files' should be a list of file names/paths"
        logger.error(err_msg)
        ValueError(err_msg)
    # Check that all mandatory files are defined
    elif any(not any(mand_file in inp_file.split("/")[-1]
                     for inp_file in inp_files) for mand_file in mand_files):
        err_msg = f"At least one of the mandatory files {mand_files} was " \
                  "not specified."
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    # Check that the defined files exist
    elif any(not os.path.isfile(inp_file) for inp_file in inp_files):
        err_msg = f"At least one of the mandatory files {mand_files} was " \
                  "not found."
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    incar = ""
    for i, inp_file in enumerate(inp_files):
        file_name = inp_file.split("/")[-1]
        if "INCAR" in file_name:
            incar = Incar.from_file(inp_file)
            incar["SYSTEM"] = proj_name+"_"+run_type

    # Builds the directory hierarchy and copies/creates the relevant files
    for c, conf in enumerate(atms_list):
        subdir = f'{run_type}/conf_{c}/'
        os.mkdir(subdir)
        for inp_file in inp_files:
            file_name = inp_file.split("/")[-1]
        if cell is not False and np.linalg.det(cell) != 0.0:
            conf.pbc = True
            conf.cell = cell
            conf.center()
        elif np.linalg.det(conf.cell) == 0:
            err_msg = "Cell is not defined"
            logger.error(err_msg)
            raise ValueError(err_msg)
        conf.write(subdir+"POSCAR", format="vasp")
        if "POTCAR" not in mand_files and potcar_dir:  # TODO make just once
            poscar_fh = open(subdir+"POSCAR", "r")
            grouped_symbols = poscar_fh.readline().split()
            poscar_fh.close()
            for symbol in grouped_symbols:
                potcar_sym_fh = open(f"{potcar_dir}/{symbol}/POTCAR", "r")
                potcar_sym_str = potcar_sym_fh.read()
                potcar_sym_fh.close()
                potcar_fh = open(subdir+"POTCAR", "a")
                potcar_fh.write(potcar_sym_str)
                potcar_fh.close()


def get_jobs_status(job_ids, stat_cmd, stat_dict):
    """Returns a list of job status for a list of job ids.

    @param job_ids: list of all jobs to be checked their status.
    @param stat_cmd: Command to check job status.
    @param stat_dict: Dictionary with pairs of job status (r, p, f) and the
        pattern it matches in the output of the stat_cmd.
    @return: list of status for every job.
    """
    from subprocess import PIPE, Popen
    status_list = []
    for job in job_ids:
        stat_msg = Popen(stat_cmd % job, shell=True,
                         stdout=PIPE).communicate()[0].decode('utf-8').strip()
        if stat_dict['r'] == stat_msg:
            status_list.append('r')
        elif stat_dict['p'] == stat_msg:
            status_list.append('p')
        elif stat_dict['f'] == stat_msg:
            status_list.append('f')
        else:
            logger.warning(f'Unrecognized job {job} status: {stat_msg}')
    return status_list


def submit_jobs(run_type, sub_cmd, sub_script, stat_cmd, stat_dict, max_jobs,
                name):
    """Submits jobs to a custom queuing system with the provided script

    @param run_type: Type of calculation. 'isolated', 'screening', 'refinement'
    @param sub_cmd: Bash command used to submit jobs.
    @param sub_script: script for the job submission.
    @param stat_cmd: Bash command to check job status.
    @param stat_dict: Dictionary with pairs of job status: r, p, f (ie. running
        pending and finished) and the pattern it matches in the output of the
        stat_cmd.
    @param max_jobs: dict: Contains the maximum number of jobs to be both
        running, pending/queued and pending+running. When the relevant maximum
        is reached no jobs more are submitted.
    @param name: name of the project.
    """
    from shutil import copy
    from time import sleep
    from subprocess import PIPE, Popen
    from src.dockonsurf.utilities import _human_key
    subm_jobs = []
    init_dir = os.getcwd()
    for conf in sorted(os.listdir(run_type), key=_human_key):
        i = conf.split('_')[1]
        while get_jobs_status(subm_jobs, stat_cmd, stat_dict).count("r") + \
                get_jobs_status(subm_jobs, stat_cmd, stat_dict).count("p") \
                >= max_jobs['rp']\
                or get_jobs_status(subm_jobs, stat_cmd, stat_dict).count("r") \
                >= max_jobs['r'] \
                or get_jobs_status(subm_jobs, stat_cmd, stat_dict).count("p") \
                >= max_jobs['p']:
            sleep(30)
        copy(sub_script, f"{run_type}/{conf}")
        os.chdir(f"{run_type}/{conf}")
        job_name = f'{name[:5]}{run_type[:3].capitalize()}{i}'
        sub_order = sub_cmd % (job_name, sub_script)
        subm_msg = Popen(sub_order, shell=True, stdout=PIPE).communicate()[0]
        job_id = None
        for word in subm_msg.decode("utf-8").split():
            try:
                job_id = int(word.replace('>', '').replace('<', ''))
                break
            except ValueError:
                continue
        subm_jobs.append(job_id)
        os.chdir(init_dir)

    logger.info('All jobs have been submitted, waiting for them to finish.')
    while not all([stat == 'f' for stat in
                   get_jobs_status(subm_jobs, stat_cmd, stat_dict)]):
        sleep(30)
    logger.info('All jobs have finished.')


def run_calc(run_type, inp_vars, atms_list):
    """Directs the calculation run/submission.

    @param run_type: Type of calculation. 'isolated', 'screening' or
    'refinement'
    @param inp_vars: Calculation parameters from input file.
    @param atms_list: List of ase.Atoms objects containing the sets of atoms
    aimed to run the calculations of.
    """
    from src.dockonsurf.utilities import check_bak

    run_types = ['isolated', 'screening', 'refinement']
    if not isinstance(run_type, str) or run_type.lower() not in run_types:
        run_type_err = f"'run_type' must be one of the following: {run_types}"
        logger.error(run_type_err)
        raise ValueError(run_type_err)

    if inp_vars['batch_q_sys']:
        logger.info(f"Running {run_type} calculation with {inp_vars['code']} on"
                    f" {inp_vars['batch_q_sys']}.")
    else:
        logger.info(f"Doing a dry run of {run_type}.")
    check_bak(run_type)
    os.mkdir(run_type)

    # Prepare directories and files for relevant code.
    input_files = {'isolated': 'isol_inp_file', 'screening': 'screen_inp_file',
                   'refinement': 'refine_inp_file', }
    if inp_vars['code'] == 'cp2k':
        prep_cp2k(inp_vars[input_files[run_type]], run_type, atms_list,
                  inp_vars['project_name'])
    elif inp_vars['code'] == "vasp":
        prep_vasp(inp_vars[input_files[run_type]], run_type, atms_list,
                  inp_vars['project_name'], inp_vars['pbc_cell'],
                  inp_vars['potcar_dir'])
    # TODO Implement code  == none
    # elif: inp_vars['code'] == 'Other codes here'

    # Submit/run Jobs
    if inp_vars['batch_q_sys'] == 'sge':
        stat_cmd = "qstat | grep %s | awk '{print $5}'"
        stat_dict = {'r': 'r', 'p': 'qw', 'f': ''}
        submit_jobs(run_type, 'qsub -N %s %s', inp_vars['subm_script'],
                    stat_cmd, stat_dict, inp_vars['max_jobs'],
                    inp_vars['project_name'])
    elif inp_vars['batch_q_sys'] == 'slurm':
        stat_cmd = "squeue | grep %s | awk '{print $5}'"
        stat_dict = {'r': 'R', 'p': 'PD', 'f': ''}
        submit_jobs(run_type, 'sbatch -J %s %s', inp_vars['subm_script'],
                    stat_cmd, stat_dict, inp_vars['max_jobs'],
                    inp_vars['project_name'])                    
    elif inp_vars['batch_q_sys'] == 'lsf':
        stat_cmd = "bjobs -w | grep %s | awk '{print $3}'"
        stat_dict = {'r': 'RUN', 'p': 'PEND', 'f': ''}
        submit_jobs(run_type, 'bsub -J %s < %s', inp_vars['subm_script'],
                    stat_cmd, stat_dict, inp_vars['max_jobs'],
                    inp_vars['project_name'])
    elif inp_vars['batch_q_sys'] == 'irene':
        stat_cmd = "ccc_mstat | grep %s | awk '{print $10}' | cut -c1"
        stat_dict = {'r': 'R', 'p': 'P', 'f': ''}
        submit_jobs(run_type, 'ccc_msub -r %s %s', inp_vars['subm_script'],
                    stat_cmd, stat_dict, inp_vars['max_jobs'],
                    inp_vars['project_name'])
    elif inp_vars['batch_q_sys'] == 'local':
        pass  # TODO implement local
    elif not inp_vars['batch_q_sys']:
        pass
    else:
        err_msg = "Unknown value for 'batch_q_sys'."
        logger.error(err_msg)
        raise ValueError(err_msg)
