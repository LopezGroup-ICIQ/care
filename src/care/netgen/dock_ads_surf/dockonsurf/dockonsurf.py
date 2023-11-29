"""
Code to systematically find the most stable geometry for molecules on surfaces
"""

from care.netgen.dock_ads_surf.dockonsurf.src.dockonsurf.dos_input import read_input
from care.netgen.dock_ads_surf.dockonsurf.src.dockonsurf.screening import run_screening

def dockonsurf(inp_vars):
    inp_vars = read_input(inp_vars)
    return run_screening(inp_vars)