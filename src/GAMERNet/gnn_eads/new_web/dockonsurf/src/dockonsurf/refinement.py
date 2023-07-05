import logging

logger = logging.getLogger('DockOnSurf')


def select_stable_confs(conf_list, energy_cutoff):
    """From a list of atomic configurations selects the most stable ones.

    Given a list of ase.Atoms configurations and an energy cutoff, selects all
    the structures that have an energy lower than, the energy of the most stable
    conformer plus the cutoff.

    @param conf_list: List of ase.Atoms objects of the conformers
    @param energy_cutoff: The maximum energy above the most stable
        configuration.
    @return: list of the the most stable configurations within the energy cutoff.
    """
    sorted_list = sorted(conf_list, key=lambda conf: conf.info['energy'])
    lowest_e = sorted_list[0].info['energy']
    return [conf for conf in sorted_list
            if conf.info['energy'] <= lowest_e + energy_cutoff]


def run_refinement(inp_vars):
    """Carries out the refinement of adsorbate-slab structures after screening.

    @param inp_vars: Calculation parameters from input file.
    """
    import os
    import numpy as np
    from src.dockonsurf.formats import collect_confs
    from src.dockonsurf.calculation import run_calc, check_finished_calcs

    logger.info('Carrying out procedures for the refinement of '
                'adsorbate-surface structures.')

    if not os.path.isdir("screening"):
        err = "'screening' directory not found. It is needed in order to carry "
        "out the refinement of structures to be adsorbed"
        logger.error(err)
        raise FileNotFoundError(err)

    finished_calcs, failed_calcs = check_finished_calcs('screening',
                                                        inp_vars['code'])
    if not finished_calcs:
        err_msg = "No calculations on 'screening' finished normally."
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    logger.info(f"Found {len(finished_calcs)} structures of "
                f"adsorbate-surface atomic configurations whose calculation"
                f" finished normally.")
    if len(failed_calcs) != 0:
        logger.warning(f"Found {len(failed_calcs)} calculations more that "
                       f"did not finish normally: {failed_calcs}. \n"
                       f"Using only the ones that finished normally: "
                       f"{finished_calcs}.")

    conf_list = collect_confs(finished_calcs, inp_vars['code'], 'screening',
                              inp_vars['special_atoms'])
    selected_confs = select_stable_confs(conf_list, inp_vars['energy_cutoff'])
    logger.info(f"Selected {len(selected_confs)} structures to carry out the"
                f" refinement")
    run_calc('refinement', inp_vars, selected_confs)
    logger.info("Finished the procedures for the refinement of "
                "adsorbate-surface structures section. ")
    if inp_vars["batch_q_sys"]:
        finished_calcs, failed_calcs = check_finished_calcs('refinement',
                                                            inp_vars['code'])
        if not finished_calcs:
            err_msg = "No calculations on 'refinement' finished normally."
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        conf_list = collect_confs(finished_calcs, inp_vars['code'],
                                  'refinement', inp_vars['special_atoms'])
        sorted_confs = select_stable_confs(conf_list, np.inf)
        logger.info(f"Most stable structure is {sorted_confs[0].info['ref']} "
                    f"with a total energy of {sorted_confs[0].info['energy']} "
                    f"eV.")
        confs_str = "\n".join([" ".join((str(conf.info['ref']), 'E =',
                                         str(conf.info['energy'] -
                                         sorted_confs[0].info['energy']),
                                         'eV'))
                               for conf in sorted_confs])
        logger.info("The relative energies, of all structures obtained at the "
                    "refinement stage, respect the most stable one "
                    f"({sorted_confs[0].info['ref']}) are:\n{confs_str}")
