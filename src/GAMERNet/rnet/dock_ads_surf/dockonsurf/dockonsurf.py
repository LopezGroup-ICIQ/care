"""
Code to systematically find the most stable geometry for molecules on surfaces
"""

from GAMERNet.rnet.dock_ads_surf.dockonsurf.src.dockonsurf.dos_input import read_input
from GAMERNet.rnet.dock_ads_surf.dockonsurf.src.dockonsurf.screening import run_screening
import pprint as pp

def dockonsurf(inp_vars):
    inp_vars = read_input(inp_vars)
    return run_screening(inp_vars)