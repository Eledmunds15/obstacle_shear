# =============================================================
# LAMMPS Monopole Input Generation
# Author: Ethan L. Edmunds
# Version: v1.0
# Description: Python script to produce input for precipitate calculations.
# Note: Dislocation is aligned along X, glide plane along Y axis.
# Command: apptainer exec 00_envs/lmp_CPU_22Jul2025.sif mpirun.openmpi -np 16 python3 02_minimize/run.py
# =============================================================

# ---------------------------
# IMPORT LIBRARIES
# ---------------------------
import os
import numpy as np
import subprocess
from mpi4py import MPI

from lammps import lammps

# =============================================================
# PATH SETTINGS
# =============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '000_data')) # Master data directory
STAGE_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '02_minimize')) # Stage data directory

OUTPUT_DIR = os.path.join(STAGE_DATA_DIR, 'output') # Output folder
DUMP_DIR = os.path.join(STAGE_DATA_DIR, 'dump') # Dump folder
LOG_DIR = os.path.join(STAGE_DATA_DIR, 'logs') # Log folder

for directory in [OUTPUT_DIR, DUMP_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

INPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '01_input', 'output')) # Input directory
INPUT_FILE = os.path.join(INPUT_DIR, 'edge_dislo_100_30_40.lmp') # Input file

POTENTIALS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_potentials')) # Potentials Directory
POTENTIAL_FILE = os.path.join(POTENTIALS_DIR, 'malerba.fs') # Potential file

# =============================================================
# SIMULATION PARAMETERS
# =============================================================

ENERGY_TOL = 1e-6 # Energy tolerance for minimization
FORCE_TOL = 1e-8 # Force tolerance for minimization
BUFF = 2

# =============================================================
# MAIN FUNCTION
# =============================================================
def main():
    # ---------- Initialize Simulation ------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    lmp = lammps()

    # ---------- Initialize Simulation ------------------------
    lmp.cmd.clear()
    lmp.cmd.log(os.path.join(LOG_DIR, 'log.lammps'))

    lmp.cmd.units('metal')
    lmp.cmd.dimension(3)
    lmp.cmd.boundary('p', 'f', 'p')

    lmp.cmd.read_data(INPUT_FILE)

    lmp.cmd.pair_style('eam/fs')
    lmp.cmd.pair_coeff('*', '*', POTENTIAL_FILE, 'Fe')

    lmp.cmd.group('all', 'type', '1')

    lmp.cmd.compute('peratom', 'all', 'pe/atom')

    lmp.cmd.minimize(ENERGY_TOL, FORCE_TOL, 1000, 10000)

    DUMP_PATH = os.path.join(DUMP_DIR, 'edge_dislo_100_30_40_dump')
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'edge_dislo_100_30_40_output.lmp')

    lmp.cmd.write_dump('all', 'custom', DUMP_PATH, 'id', 'x', 'y', 'z', 'c_peratom')
    lmp.cmd.write_data(OUTPUT_PATH)

    return None

# =============================================================
# FUNCTIONS
# =============================================================

# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    main()
