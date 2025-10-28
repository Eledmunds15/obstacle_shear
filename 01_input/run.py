# =============================================================
# LAMMPS Monopole Input Generation
# Author: Ethan L. Edmunds
# Version: v1.0
# Description: Python script to produce input for precipitate calculations.
# Note: Dislocation is aligned along Z, glide plane along X axis, climb plane is Y axis.
# Command: apptainer exec 00_envs/lmp_CPU_22Jul2025.sif python3 01_input/run.py
# =============================================================

# ---------------------------
# IMPORT LIBRARIES
# ---------------------------
import os
import numpy as np
import subprocess

from matscipy.calculators.eam import EAM
from matscipy.dislocation import get_elastic_constants

# =============================================================
# PATH SETTINGS
# =============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '000_data')) # Master data directory
STAGE_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '01_input')) # Stage data directory

OUTPUT_DIR = os.path.join(STAGE_DATA_DIR, 'output') # Output folder
DUMP_DIR = os.path.join(STAGE_DATA_DIR, 'dump') # Dump folder
LOG_DIR = os.path.join(STAGE_DATA_DIR, 'logs') # Log folder

for directory in [OUTPUT_DIR, DUMP_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

POTENTIALS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_potentials')) # Potentials Directory
POTENTIAL_FILE = os.path.join(POTENTIALS_DIR, 'malerba.fs') # Potential file

# =============================================================
# SIMULATION PARAMETERS
# =============================================================
# Lattice dimensions (angstrom)
X_MIN = 100  # LENGTH ALONG X
Y_MIN = 30  # LENGTH ALONG Y
Z_MIN = 40  # LENGTH ALONG Z

# =============================================================
# MAIN FUNCTION
# =============================================================
def main():
    # ---------------------------
    # Load EAM potential
    # ---------------------------
    eam_calc = EAM(POTENTIAL_FILE)

    # Get lattice constant and elastic constants for Fe
    alat, C11, C12, C44 = get_elastic_constants(calculator=eam_calc, symbol="Fe", verbose=True)

    # ---------------------------
    # Define dislocation
    # ---------------------------

    subprocess.run(['atomsk', '--create', 'bcc', str(alat), 'Fe', 'orient', '[111]', '[1-10]', '[11-2]', 'Fe_unitcell.cfg'])

    subprocess.run(['atomsk', 'Fe_unitcell.cfg', '-duplicate', str(X_MIN), str(Y_MIN), str(Z_MIN), '-deform', 'X', str(0.5/X_MIN), '0.0', 'bottom.cfg'])

    subprocess.run(['atomsk', 'Fe_unitcell.cfg', '-duplicate', str(X_MIN+1), str(Y_MIN), str(Z_MIN), '-deform', 'X', str(-0.5/(X_MIN+1)), '0.0', 'top.cfg'])

    subprocess.run(['atomsk', '--merge', 'Y', '2', 'bottom.cfg', 'top.cfg', f'edge_dislo_{X_MIN}_{Y_MIN}_{Z_MIN}.lmp'])

# =============================================================
# FUNCTIONS
# =============================================================

# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    main()
