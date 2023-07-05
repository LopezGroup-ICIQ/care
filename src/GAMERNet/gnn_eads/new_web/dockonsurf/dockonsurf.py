#!/usr/bin/env python3
"""
Code to systematically find the most stable geometry for molecules on surfaces
"""

from src.dockonsurf.dos_input import read_input
from src.dockonsurf.screening import run_screening


def dockonsurf(file):

    inp_vars = read_input(file)

    surf_ads_list = run_screening(inp_vars)

    return surf_ads_list