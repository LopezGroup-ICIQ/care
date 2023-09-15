"""
Code to systematically find the most stable geometry for molecules on surfaces
"""

from GAMERNet.rnet.dock_ads_surf.dockonsurf.src.dockonsurf.dos_input import read_input
from GAMERNet.rnet.dock_ads_surf.dockonsurf.src.dockonsurf.screening import run_screening


def dockonsurf(file):
    inp_vars = read_input(file)
    return run_screening(inp_vars)