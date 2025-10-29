# ---------------------------
# IMPORT LIBRARIES
# ---------------------------

import os
import re
from mpi4py import MPI

from ovito.io import import_file, export_file
from ovito.modifiers import DislocationAnalysisModifier, WignerSeitzAnalysisModifier, DeleteSelectedModifier, InvertSelectionModifier
from ovito.pipeline import FileSource

# =============================================================
# PATH SETTINGS
# =============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '000_data')) # Master data directory
STAGE_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '03_pin_dislo')) # Stage data directory

DXA_DIR = os.path.join(STAGE_DATA_DIR, 'dxa') # File with DXA analysis
DXA_SUMMARY_DIR = os.path.join(STAGE_DATA_DIR, 'dxa_summary') # File with DXA summary files
DXA_ATOMS_DIR = os.path.join(STAGE_DATA_DIR, 'dxa_atoms') # File with the atoms extracted by DXA
WS_VAC_DIR = os.path.join(STAGE_DATA_DIR, 'wigner_seitz_vacs') # File with wigner seitz analysis files
WS_SIA_DIR = os.path.join(STAGE_DATA_DIR, 'wigner_seitz_sias') # File with wigner seitz analysis files

for directory in [DXA_DIR, DXA_SUMMARY_DIR, DXA_ATOMS_DIR, WS_VAC_DIR, WS_SIA_DIR]:
    os.makedirs(directory, exist_ok=True)

REFERENCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '02_minimize', 'dump')) # Input directory
REFERENCE_FILE = os.path.join(REFERENCE_DIR, 'edge_dislo_100_60_40_dump') # Input file

DATA_DIR = os.path.abspath(os.path.join(STAGE_DATA_DIR, 'dump'))

# =============================================================
# MAIN FUNCTION
# =============================================================

def main():
    #--- INITIALISE MPI ---#
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #--- INITIALISE VARIABLE ON ALL RANKS ---#
    dump_files = None

    #--- Get files ---#
    if rank == 0: dump_files = get_filenames(DATA_DIR)

    #--- BROADCAST AND DISTRIBUTE WORK ---#
    dump_files = comm.bcast(dump_files, root=0)

    # Each rank gets only its share of files to process
    start, end = split_indexes(len(dump_files), rank, size)

    print(f"Rank {rank} of size {size} processing files from {start} to {end}", flush=True)

    #--- PROCESS FILES ---#
    process_file(dump_files[start:end])

    comm.Barrier()

    if rank == 0: print("Successfully processed all files...")
    
    return None

# --------------------------- ANALYSIS ---------------------------#

def process_file(dump_chunk):

    input_paths = [os.path.join(DATA_DIR, dump_file) for dump_file in dump_chunk]

    for frame in input_paths:

        pipeline = import_file(frame)
        data = pipeline.compute()

        performDXA(data)
        performWS(data)
        
        print(f"Successfully processed frame {frame}...", flush=True)

def performDXA(data):

    dxaModifier = DislocationAnalysisModifier(input_crystal_structure=DislocationAnalysisModifier.Lattice.BCC)
    # Ignore defect mesh when exporting ???
    data.apply(dxaModifier) # Run DXA
    
    # Select normal sites
    expModifier = ExpressionSelectionModifier(expression = 'Cluster == 1')
    data.apply(expModifier)

    # Delete Selected
    delModifier = DeleteSelectedModifier()
    data.apply(delModifier)

    timestep = data.attributes['Timestep']

    export_file(data, os.path.join(DXA_DIR, f'dxa_{int(timestep)}'), "ca", export_mesh=False)

    export_file(data, os.path.join(DXA_ATOMS_DIR, f'dxa_atoms_{int(timestep)}'), "lammps/dump",
            columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z", "c_peratom", "Cluster"])

    print(f"DXA for timestep {timestep} complete...", flush=True)

    return None

def performWS(data):

    # Modifier Definition
    wsModifier = WignerSeitzAnalysisModifier()
    wsModifier.reference = FileSource()
    wsModifier.reference.load(REFERENCE_FILE)

    data.apply(wsModifier)

    timestep = data.attributes['Timestep']

    """print(f"Data in data: {data.particles}")
    print(f"Number of total particles {data.particles.count}")"""

    data_vac = data.clone()
    data_sia = data.clone()

    """print(f"\nVacancies")
    print(f"Data in data: {data_vac.particles}")
    print(f"Number of total particles {data_vac.particles.count}")

    print(f"\nSelf-interstitials")
    print(f"Data in data: {data_sia.particles}")
    print(f"Number of total particles {data_sia.particles.count}")"""

    occupancies = data.particles['Occupancy']

    selection_vac = data_vac.particles_.create_property('Selection')
    selection_sia = data_sia.particles_.create_property('Selection')

    selection_vac[...] = (occupancies == 0)
    selection_sia[...] = (occupancies == 2)

    """print(f"\nNumber of vacancies = {np.sum(selection_vac)}")
    print(f"Number of self-interstitial = {np.sum(selection_sia)}")"""

    data_vac.apply(InvertSelectionModifier())
    data_sia.apply(InvertSelectionModifier())

    data_vac.apply(DeleteSelectedModifier())
    data_sia.apply(DeleteSelectedModifier())

    """print(data_vac.particles.count)
    print(data_sia.particles.count)"""

    # Export the file
    export_file(
            data_vac,
            os.path.join(WS_VAC_DIR, f'ws_vac_{timestep}'),
            "lammps/dump",
            columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z"],
        )

    export_file(
            data_sia,
            os.path.join(WS_SIA_DIR, f'ws_vac_{timestep}'),
            "lammps/dump",
            columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z"],
    )

    print(f"DXA for timestep {timestep} complete...", flush=True)

    return None

# --------------------------- UTILITIES ---------------------------#

def view_information(data):
    
    print('')
    print("Available particle properties:")
    for prop in data.particles.keys():
        print(f"  - {prop}")

    print("\nAvailable global attributes:")
    for attr in data.attributes.keys():
        print(f"  - {attr}")
    print('')

def get_filenames(dir_path):
    """Returns a naturally sorted list of filenames (not paths) in the given directory."""
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    return sorted(files, key=natural_sort_key)

def natural_sort_key(s):
    # Split the string into digit and non-digit parts, convert digits to int
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def split_indexes(n_files, rank, size):
    """Split n_files into contiguous chunks of indexes for each rank."""
    chunk_size = n_files // size
    remainder = n_files % size

    if rank < remainder:
        start = rank * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = rank * chunk_size + remainder
        end = start + chunk_size

    return [start, end]

# --------------------------- ENTRY POINT ---------------------------#

if __name__ == "__main__":

        main()