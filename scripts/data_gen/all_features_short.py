"""
Script for generating tomograms simulating all features available
monomers
    Input:
        - Number of tomograms to simulate
        - Tomogram dimensions parameter
        - Tomogram maximum occupancy
        - Features to simulate:
            + Membranes
            + Polymers:
                + Helicoidal fibers
                + Globular protein clusters
        - 3D reconstruction paramaters
    Output:
        - The simulated density maps
        - The 3D reconstructed tomograms
        - Micrograph stacks
        - Polydata files
        - STAR file mapping particle coordinates and orientations with tomograms
"""

__author__ = "Antonio Martinez-Sanchez"

import sys
import csv
import time
import random
import tarfile
import math
import numpy as np
from polnet.utils import *
from polnet import lio
from polnet import tem
from polnet import poly as pp
from polnet.network import (
    NetSAWLC,
    NetSAWLCInter,
    NetHelixFiber,
    NetHelixFiberB,
)
from polnet.polymer import FiberUnitSDimer, MTUnit, MB_DOMAIN_FIELD_STR
from polnet.stomo import (
    MmerFile,
    MbFile,
    SynthTomo,
    SetTomos,
    HelixFile,
    MTFile,
    ActinFile,
    MmerMbFile,
)
from polnet.lrandom import (
    EllipGen,
    SphGen,
    TorGen,
    PGenHelixFiberB,
    PGenHelixFiber,
    SGenUniform,
    SGenProp,
    OccGen,
)
from polnet.membrane import SetMembranes
from tqdm import tqdm
import pandas as pd
##### Logging configuration
import os
import logging
import sys

class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ""

    def write(self, message):
        # Buffer the message until a newline is encountered
        self.buffer += message
        if "\n" in self.buffer:
            self.level(self.buffer.strip())  # Write the buffered message to the log
            self.buffer = ""  # Clear the buffer

    def flush(self):
        if self.buffer:  # Write any remaining buffered content
            self.level(self.buffer.strip())
            self.buffer = ""
# Set up logging

# Define a hard-coded output directory
OUT_DIR = "/mnt/lustre-emmy-hdd/usr/u15206/simulation_dir/all_v_czii_1"
os.makedirs(OUT_DIR, exist_ok=True)
# Let the log files include the job number (if available)
LOG_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
job_id = os.environ.get("SLURM_JOB_ID", "default")
log_path = os.path.join(LOG_DIR, f"simulation-output_{job_id}.log")
error_log_path = os.path.join(LOG_DIR, f"simulation_{job_id}_error.log")

# Configure logging to write to the regular log file and the error log file
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

# Regular log handler
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(logging.Formatter("%(message)s"))  # Only log the message
logger.addHandler(file_handler)

# Error log handler
error_handler = logging.FileHandler(error_log_path)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(message)s"))  # Only log the message
logger.addHandler(error_handler)

logger.info("Logging initialized. Log file: %s", log_path)
logger.info("Error logging initialized. Error log file: %s", error_log_path)

sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)
##### Input parameters

# Common tomogram settings
#ROOT_PATH = os.path.realpath(os.getcwd() + "/../../data")
ROOT_PATH = "/user/yusufberk.oruc/u15206/polnet/data_new_1"
ROOT_PATH_ACTIN = "/user/yusufberk.oruc/u15206/polnet/data_new"
ROOT_PATH_MEMBRANE = "/user/yusufberk.oruc/u15206/polnet/data_new"

NTOMOS = 30  # 12
VOI_SHAPE = (
    1000,
    1000,
    250,
)  # (1000, 1000, 400) # (400, 400, 236) # vx or a path to a mask (1-foreground, 0-background) tomogram
czII_challenge_data = True
if czII_challenge_data:
    VOI_SHAPE = (630,630,184)
VOI_OFFS = (
    (4, 996),
    (4, 996),
    (4, 246),
)  # ((4,396), (4,396), (4,232)) # ((4,1852), (4,1852), (32,432)) # ((4,1852), (4,1852), (4,232)) # vx
VOI_VSIZE = 10  # 2.2 # A/vx
MMER_TRIES = 1
PMER_TRIES = 1000

# Lists with the features to simulate
# Membranes
MEMBRANES_LIST = [
    "in_mbs/sphere.mbs",
    "in_mbs/ellipse.mbs",
    "in_mbs/toroid.mbs",
]

# Helicoidal proteins
HELIX_LIST = ["in_helix/mt.hns", "in_helix/actin.hns"]

# Proteins
PROTEINS_LIST = [
    "in_10A/4v4r_10A.pns",
    "in_10A/3j9i_10A.pns",
    "in_10A/4v4r_50S_10A.pns",
    "in_10A/4v4r_30S_10A.pns",
    "in_10A/6utj_10A.pns",
    "in_10A/5mrc_10A.pns",
    "in_10A/4v7r_10A.pns",
    "in_10A/2uv8_10A.pns",
    "in_10A/4v94_10A.pns",
    "in_10A/4cr2_10A.pns",
    "in_10A/3qm1_10A.pns",
    "in_10A/3h84_10A.pns",
    "in_10A/3gl1_10A.pns",
    "in_10A/3d2f_10A.pns",
    "in_10A/3cf3_10A.pns",
    "in_10A/2cg9_10A.pns",
    "in_10A/1u6g_10A.pns",
    "in_10A/1s3x_10A.pns",
    "in_10A/1qvr_10A.pns",
    "in_10A/1bxn_10A.pns",
]
# Membrane proteins
MB_PROTEINS_LIST = [
    "in_10A/mb_6rd4_10A.pms",
    "in_10A/mb_5wek_10A.pms",
    "in_10A/mb_4pe5_10A.pms",
    "in_10A/mb_5ide_10A.pms",
    "in_10A/mb_5gjv_10A.pms",
    "in_10A/mb_5kxi_10A.pms",
    "in_10A/mb_5tj6_10A.pms",
    "in_10A/mb_5tqq_10A.pms",
    "in_10A/mb_5vai_10A.pms",
]
# New proteins list
NEW_PROTEINS_LIST = [
    "in_10A/6drv_10A.pns",
    "in_10A/6n4v_10A.pns",
    "in_10A/6qzp_10A.pns",
    "in_10A/7n4y_10A.pns",
    "in_10A/8cpv_10A.pns",
    "in_10A/8vaf_10A.pns",
    "in_10A/1fa2_10A.pns"]

PROP_LIST_RAW= np.array([5, 6, 6, 80 ,13,47 ,1])

new_proteins = False
only_new_proteins = True
not_use_membrane_proteins = True

if new_proteins:
    PROTEINS_LIST += NEW_PROTEINS_LIST
if only_new_proteins:
    PROTEINS_LIST = NEW_PROTEINS_LIST
    # PROTEINS_LIST = NEW_PROTEINS_LIST
if not_use_membrane_proteins:
    MB_PROTEINS_LIST = []
# Proportions list, specifies the proportion for each protein, this proportion is tried to be achieved but no guaranteed
# The toal sum of this list must be 1
PROP_LIST_Flag = False # True # False

 # [.4, .6]
if  PROP_LIST_Flag:
    PROP_LIST = PROP_LIST_RAW / np.sum(PROP_LIST_RAW)  
else:
    PROP_LIST = None

if PROP_LIST is not None:
    assert sum(PROP_LIST) == 1

SURF_DEC = 0.9  # Target reduction factor for surface decimation (default None)

# Reconstruction tomograms
TILT_ANGS = np.arange(
    -60, 60, 3
)  # range(-90, 91, 3) # at MPI-B IMOD only works for ranges
DETECTOR_SNR = [1.0, 2.0]  # 0.2 # [.15, .25]
MALIGN_MN = 1
MALIGN_MX = 1.5
MALIGN_SG = 0.2

# OUTPUT FILES
"""OUT_DIR = os.path.realpath(
    ROOT_PATH + "/data_generated/all_v11"
)  # '/out_all_tomos_9-10' # '/only_actin' # '/out_rotations'"""


os.makedirs(OUT_DIR, exist_ok=True)

TEM_DIR = OUT_DIR + "/tem"
TOMOS_DIR = OUT_DIR + "/tomos"
os.makedirs(TOMOS_DIR, exist_ok=True)
os.makedirs(TEM_DIR, exist_ok=True)

# OUTPUT LABELS
LBL_MB = 1
LBL_AC = 2
LBL_MT = 3
LBL_CP = 4
LBL_MP = 5
# LBL_BR = 6

if "--print-parameters" in sys.argv:
    print("=== Initial Parameters and Settings ===")
    print("ROOT_PATH:", ROOT_PATH)
    print("ROOT_PATH_ACTIN:", ROOT_PATH_ACTIN)
    print("ROOT_PATH_MEMBRANE:", ROOT_PATH_MEMBRANE)
    print("NTOMOS:", NTOMOS)
    print("VOI_SHAPE:", VOI_SHAPE)
    print("VOI_OFFS:", VOI_OFFS)
    print("VOI_VSIZE:", VOI_VSIZE)
    print("MMER_TRIES:", MMER_TRIES)
    print("PMER_TRIES:", PMER_TRIES)
    print("czII_challenge_data", czII_challenge_data)
    print("not_use_membranes:", not_use_membrane_proteins)
    print("new_proteins:", new_proteins)
    print("only_new_proteins:", only_new_proteins)
    print("\n--- Feature Lists ---")
    print("MEMBRANES_LIST:", MEMBRANES_LIST)
    print("HELIX_LIST:", HELIX_LIST)
    print("PROTEINS_LIST:", PROTEINS_LIST)
    print("NEW_PROTEINS_LIST:", NEW_PROTEINS_LIST)
    print("MB_PROTEINS_LIST:", MB_PROTEINS_LIST)
    print("PROP_LIST:", PROP_LIST)
    print("PROP_LIST_Flag:", PROP_LIST_Flag)
    print("SURF_DEC:", SURF_DEC)
    print("\n--- Reconstruction Settings ---")
    print("TILT_ANGS:", TILT_ANGS)
    print("DETECTOR_SNR:", DETECTOR_SNR)
    print("MALIGN_MN:", MALIGN_MN)
    print("MALIGN_MX:", MALIGN_MX)
    print("MALIGN_SG:", MALIGN_SG)
    print("\n--- Output Directories ---")
    print("OUT_DIR:", OUT_DIR)
    print("TEM_DIR:", TEM_DIR)
    print("TOMOS_DIR:", TOMOS_DIR)
    print("error_log_path:", error_log_path)
    print("log_path:", log_path)
    print("=== End of Parameters ===")
    sys.stdout.flush()  # Ensure output is written
    
##### Functions
def save_input_files(root_path, output_file, exclude_dirs=None):
    """
    Save all files from the ROOT_PATH into a single archive, excluding specified directories.

    :param root_path: The root directory containing the input files.
    :param output_file: The path to the output archive file.
    :param exclude_dirs: A list of directory names to exclude (e.g., ["templates"]).
    """
    exclude_dirs = exclude_dirs or []
    with tarfile.open(output_file, "w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Exclude specified directories
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                arcname = os.path.relpath(file_path, root_path)  # Preserve relative paths
                tar.add(file_path, arcname=arcname)
    print(f"Input files saved to {output_file}")

def display_sample_files_with_content(root_path, exclude_dirs=None, specific_files=None):
    """
    Display one file from each subdirectory in the root path and print its content, prioritizing specific files.

    :param root_path: The root directory to search.
    :param exclude_dirs: A list of directory names to exclude (e.g., ["templates"]).
    :param specific_files: A dictionary where keys are directories and values are specific filenames to display.
    """
    exclude_dirs = exclude_dirs or []
    specific_files = specific_files or {}
    print("Sample files and their content from each directory:")
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Exclude specified directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        # Exclude hidden files (e.g., .DS_Store)
        filenames = [f for f in filenames if not f.startswith(".")]
        if filenames:
            # Check if a specific file is requested for this directory
            relative_dir = os.path.relpath(dirpath, root_path)
            if relative_dir in specific_files and specific_files[relative_dir] in filenames:
                sample_file = os.path.join(dirpath, specific_files[relative_dir])
            else:
                # Default to the first file in the directory
                sample_file = os.path.join(dirpath, filenames[0])
            
            print(f"\nFile: {sample_file}")
            # Display the content of the file
            try:
                with open(sample_file, "r") as f:
                    content = f.read()
                    print("Content:")
                    print(content[:500])  # Print the first 500 characters to avoid excessive output
            except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError) as e:
                print(f"Could not read file {sample_file}: {e}")

def display_statistics_from_csv(csv_file):
    """
    Reads the tomos_motif_list.csv file and displays statistics.

    :param csv_file: Path to the CSV file containing tomogram data.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file, delimiter="\t")
        
        # Display general information
        print("\n=== Statistics from tomos_motif_list.csv ===")
        print(f"Total number of particles: {len(df)}")
        
        # Example statistics: Count occurrences of each motif type
        if "Code" in df.columns:
            motif_counts = df["Code"].value_counts()
            print("\nMotif counts:")
            print(motif_counts)
        
        # Example statistics: Count occurrences of each label
        if "Label" in df.columns:
            label_counts = df["Label"].value_counts()
            print("\nLabel counts:")
            print(label_counts)
        
        print("\n=== End of Statistics ===")
    except Exception as e:
        print(f"Error reading or processing the CSV file: {e}")

######preview the input files
# Display one file from each directory and its content, excluding "templates"
# Display one file from each directory and its content, prioritizing specific files
specific_files = {
    "in_10A": "6qzp_10A.pns",  # Specify the file to display for the in_10A directory
}
display_sample_files_with_content(ROOT_PATH, exclude_dirs=["templates","in_mbs","in_helix"], specific_files=specific_files)
display_sample_files_with_content(ROOT_PATH_ACTIN, exclude_dirs=["templates","in_10A"],specific_files=specific_files)
# define the output archive path
OUTPUT_ARCHIVE = os.path.join(OUT_DIR, "input_files.tar.gz")
OUTPUT_ARCHIVE_ACTIN = os.path.join(OUT_DIR, "input_files_mb_actin.tar.gz")

# Save all files except those in the "templates" directory
save_input_files(ROOT_PATH, OUTPUT_ARCHIVE, exclude_dirs=["templates","in_mbs","in_helix"])
save_input_files(ROOT_PATH_ACTIN, OUTPUT_ARCHIVE_ACTIN, exclude_dirs=["templates","in_10A"])


##### Main procedure
set_stomos = SetTomos()
vx_um3 = (VOI_VSIZE * 1e-4) ** 3

# Preparing intermediate directories
clean_dir(TEM_DIR)
clean_dir(TOMOS_DIR)

# Save labels table
unit_lbl = 1
header_lbl_tab = ["MODEL", "LABEL"]
with open(OUT_DIR + "/labels_table.csv", "w") as file_csv:
    writer_csv = csv.DictWriter(
        file_csv, fieldnames=header_lbl_tab, delimiter="\t"
    )
    writer_csv.writeheader()
    for i in range(len(MEMBRANES_LIST)):
        writer_csv.writerow(
            {header_lbl_tab[0]: MEMBRANES_LIST[i], header_lbl_tab[1]: unit_lbl}
        )
        unit_lbl += 1
    for i in range(len(HELIX_LIST)):
        writer_csv.writerow(
            {header_lbl_tab[0]: HELIX_LIST[i], header_lbl_tab[1]: unit_lbl}
        )
        unit_lbl += 1
    for i in range(len(PROTEINS_LIST)):
        writer_csv.writerow(
            {header_lbl_tab[0]: PROTEINS_LIST[i], header_lbl_tab[1]: unit_lbl}
        )
        unit_lbl += 1
    for i in range(len(MB_PROTEINS_LIST)):
        writer_csv.writerow(
            {
                header_lbl_tab[0]: MB_PROTEINS_LIST[i],
                header_lbl_tab[1]: unit_lbl,
            }
        )
        unit_lbl += 1


# Loop for tomograms
for tomod_id in tqdm(range(NTOMOS),desc="Generating tomograms"):

    print("GENERATING TOMOGRAM NUMBER:", tomod_id)
    hold_time = time.time()

    # Generate the VOI and tomogram density
    if isinstance(VOI_SHAPE, str):
        voi = lio.load_mrc(VOI_SHAPE) > 0
        voi_off = np.zeros(shape=voi.shape, dtype=bool)
        voi_off[
            VOI_OFFS[0][0] : VOI_OFFS[0][1],
            VOI_OFFS[1][0] : VOI_OFFS[1][1],
            VOI_OFFS[2][0] : VOI_OFFS[2][1],
        ] = True
        voi = np.logical_and(voi, voi_off)
        del voi_off
    else:
        voi = np.zeros(shape=VOI_SHAPE, dtype=bool)
        voi[
            VOI_OFFS[0][0] : VOI_OFFS[0][1],
            VOI_OFFS[1][0] : VOI_OFFS[1][1],
            VOI_OFFS[2][0] : VOI_OFFS[2][1],
        ] = True
        voi_inital_invert = np.invert(voi)
    bg_voi = voi.copy()
    voi_voxels = voi.sum()
    tomo_lbls = np.zeros(shape=VOI_SHAPE, dtype=np.float32)
    tomo_den = np.zeros(shape=voi.shape, dtype=np.float32)
    synth_tomo = SynthTomo()
    poly_vtp, mbs_vtp, skel_vtp = None, None, None
    entity_id = 1
    mb_voxels, ac_voxels, mt_voxels, cp_voxels, mp_voxels = 0, 0, 0, 0, 0
    set_mbs = None

    # Membranes loop
    count_mbs, hold_den = 0, None
    for p_id, p_file in tqdm(enumerate(MEMBRANES_LIST),desc="Generating membranes"):

        print("\tPROCESSING FILE:", p_file)

        # Loading the membrane file
        memb = MbFile()
        memb.load_mb_file(ROOT_PATH_MEMBRANE + "/" + p_file)

        # Generating the occupancy
        hold_occ = memb.get_occ()
        if hasattr(hold_occ, "__len__"):
            hold_occ = OccGen(hold_occ).gen_occupancy()

        # Membrane random generation by type
        param_rg = (
            memb.get_min_rad(),
            math.sqrt(3) * max(VOI_SHAPE) * VOI_VSIZE,
            memb.get_max_ecc(),
        )
        if memb.get_type() == "sphere":
            mb_sph_generator = SphGen(radius_rg=(param_rg[0], param_rg[1]))
            set_mbs = SetMembranes(
                voi,
                VOI_VSIZE,
                mb_sph_generator,
                param_rg,
                memb.get_thick_rg(),
                memb.get_layer_s_rg(),
                hold_occ,
                memb.get_over_tol(),
                bg_voi=bg_voi,
            )
            set_mbs.build_set(verbosity=True)
            hold_den = set_mbs.get_tomo()
            if memb.get_den_cf_rg() is not None:
                hold_den *= mb_sph_generator.gen_den_cf(
                    memb.get_den_cf_rg()[0], memb.get_den_cf_rg()[1]
                )
        elif memb.get_type() == "ellipse":
            mb_ellip_generator = EllipGen(
                radius_rg=param_rg[:2], max_ecc=param_rg[2]
            )
            set_mbs = SetMembranes(
                voi,
                VOI_VSIZE,
                mb_ellip_generator,
                param_rg,
                memb.get_thick_rg(),
                memb.get_layer_s_rg(),
                hold_occ,
                memb.get_over_tol(),
                bg_voi=bg_voi,
            )
            set_mbs.build_set(verbosity=True)
            hold_den = set_mbs.get_tomo()
            if memb.get_den_cf_rg() is not None:
                hold_den *= mb_ellip_generator.gen_den_cf(
                    memb.get_den_cf_rg()[0], memb.get_den_cf_rg()[1]
                )
        elif memb.get_type() == "toroid":
            mb_tor_generator = TorGen(radius_rg=(param_rg[0], param_rg[1]))
            set_mbs = SetMembranes(
                voi,
                VOI_VSIZE,
                mb_tor_generator,
                param_rg,
                memb.get_thick_rg(),
                memb.get_layer_s_rg(),
                hold_occ,
                memb.get_over_tol(),
                bg_voi=bg_voi,
            )
            set_mbs.build_set(verbosity=True)
            hold_den = set_mbs.get_tomo()
            if memb.get_den_cf_rg() is not None:
                hold_den *= mb_tor_generator.gen_den_cf(
                    memb.get_den_cf_rg()[0], memb.get_den_cf_rg()[1]
                )
        else:
            print("ERROR: Membrane type", memb.get_type(), "not recognized!")
            sys.exit()

        # Density tomogram updating
        voi = set_mbs.get_voi()
        mb_mask = set_mbs.get_tomo() > 0
        mb_mask[voi_inital_invert] = False
        tomo_lbls[mb_mask] = entity_id
        count_mbs += set_mbs.get_num_mbs()
        mb_voxels += (tomo_lbls == entity_id).sum()
        tomo_den = np.maximum(tomo_den, hold_den)
        hold_vtp = set_mbs.get_vtp()
        pp.add_label_to_poly(hold_vtp, entity_id, "Entity", mode="both")
        pp.add_label_to_poly(hold_vtp, LBL_MB, "Type", mode="both")
        if poly_vtp is None:
            poly_vtp = hold_vtp
            skel_vtp = hold_vtp
        else:
            poly_vtp = pp.merge_polys(poly_vtp, hold_vtp)
            skel_vtp = pp.merge_polys(skel_vtp, hold_vtp)
        synth_tomo.add_set_mbs(set_mbs, "Membrane", entity_id, memb.get_type())
        entity_id += 1

    # Get membranes poly
    if set_mbs is not None:
        mbs_vtp = vtk.vtkPolyData()
        mbs_vtp.DeepCopy(poly_vtp)

    # Loop for Helicoidal structures
    count_actins, count_mts = 0, 0
    for p_id, p_file in tqdm(enumerate(HELIX_LIST),desc="Generating helicoidal structures (actin)"):

        print("\tPROCESSING FILE:", p_file)

        # Loading the helix file
        helix = HelixFile()
        helix.load_hx_file(ROOT_PATH_ACTIN + "/" + p_file)

        # Generating the occupancy
        hold_occ = helix.get_occ()
        if hasattr(hold_occ, "__len__"):
            hold_occ = OccGen(hold_occ).gen_occupancy()

        # Helicoida random generation by type
        if helix.get_type() == "mt":

            helix = MTFile()
            helix.load_mt_file(ROOT_PATH_ACTIN + "/" + p_file)
            # Fiber unit generation
            funit = MTUnit(
                helix.get_mmer_rad(),
                helix.get_rad(),
                helix.get_nunits(),
                VOI_VSIZE,
            )
            model_svol, model_surf = funit.get_tomo(), funit.get_vtp()
            # Helix Fiber parameters model
            pol_generator = PGenHelixFiber()
            # Network generation
            net_helix = NetHelixFiber(
                voi,
                VOI_VSIZE,
                helix.get_l() * helix.get_mmer_rad() * 2,
                model_surf,
                pol_generator,
                hold_occ,
                helix.get_min_p_len(),
                helix.get_hp_len(),
                helix.get_mz_len(),
                helix.get_mz_len_f(),
                helix.get_over_tol(),
                (helix.get_rad() + 0.5 * helix.get_mmer_rad()) * 2.4,
            )
            if helix.get_min_nmmer() is not None:
                net_helix.set_min_nmmer(helix.get_min_nmmer())
            net_helix.build_network()
        elif helix.get_type() == "actin":
            helix = ActinFile()
            helix.load_ac_file(ROOT_PATH_ACTIN + "/" + p_file)
            # Fiber unit generation
            funit = FiberUnitSDimer(helix.get_mmer_rad(), VOI_VSIZE)
            model_svol, model_surf = funit.get_tomo(), funit.get_vtp()
            # Helix Fiber parameters model
            pol_generator = PGenHelixFiberB()
            # Network generation
            net_helix = NetHelixFiberB(
                voi,
                VOI_VSIZE,
                helix.get_l() * helix.get_mmer_rad() * 2,
                model_surf,
                pol_generator,
                hold_occ,
                helix.get_min_p_len(),
                helix.get_hp_len(),
                helix.get_mz_len(),
                helix.get_mz_len_f(),
                helix.get_bprop(),
                helix.get_p_branch(),
                helix.get_over_tol(),
            )
            if helix.get_min_nmmer() is not None:
                net_helix.set_min_nmmer(helix.get_min_nmmer())
            net_helix.build_network()
            # Geting branches poly
            br_vtp = pp.points_to_poly_spheres(
                points=[
                    [0, 0, 0],
                ],
                rad=helix.get_mmer_rad(),
            )
            lio.save_vtp(
                net_helix.get_branches_vtp(shape_vtp=br_vtp),
                TOMOS_DIR + "/poly_br_" + str(tomod_id) + ".vtp",
            )
        else:
            print("ERROR: Helicoidal type", helix.get_type(), "not recognized!")
            sys.exit()

        # # DEBUG
        # lio.save_vtp(funit.get_vtp(), ROOT_PATH + '/hold_funit.vtp')
        # lio.save_vtp(net_helix.get_vtp(), ROOT_PATH + '/hold.vtp')

        # Density tomogram updating
        # voi = net_helix.get_voi()
        # tomo_den = np.maximum(tomo_den, net_helix.get_tomo())
        model_mask = model_svol < 0.05
        # off = .5 * np.asarray(model_svol.shape) - center
        net_helix.insert_density_svol(
            model_mask, voi, VOI_VSIZE, merge="min", off_svol=None
        )
        if helix.get_den_cf_rg() is None:
            cte_val = 1
        else:
            cte_val = pol_generator.gen_den_cf(
                helix.get_den_cf_rg()[0], helix.get_den_cf_rg()[1]
            )
        net_helix.insert_density_svol(
            model_svol * cte_val, tomo_den, VOI_VSIZE, merge="max"
        )
        hold_lbls = np.zeros(shape=tomo_lbls.shape, dtype=np.float32)
        net_helix.insert_density_svol(
            np.invert(model_mask), hold_lbls, VOI_VSIZE, merge="max"
        )
        tomo_lbls[hold_lbls > 0] = entity_id
        # lio.write_mrc(hold_lbls.astype(np.float32), '/fs/pool/pool-lucic2/antonio/polnet/riboprot/hold.mrc')
        hold_vtp = net_helix.get_vtp()
        hold_skel_vtp = net_helix.get_skel()
        pp.add_label_to_poly(hold_vtp, entity_id, "Entity", mode="both")
        pp.add_label_to_poly(hold_skel_vtp, entity_id, "Entity", mode="both")
        if helix.get_type() == "mt":
            pp.add_label_to_poly(hold_vtp, LBL_MT, "Type", mode="both")
            pp.add_label_to_poly(hold_skel_vtp, LBL_MT, "Type", mode="both")
            count_mts += net_helix.get_num_pmers()
            mt_voxels += (tomo_lbls == entity_id).sum()
        elif helix.get_type() == "actin":
            pp.add_label_to_poly(hold_vtp, LBL_AC, "Type", mode="both")
            pp.add_label_to_poly(hold_skel_vtp, LBL_AC, "Type", mode="both")
            count_actins += net_helix.get_num_pmers()
            ac_voxels += (tomo_lbls == entity_id).sum()
        if poly_vtp is None:
            poly_vtp = hold_vtp
            skel_vtp = hold_skel_vtp
        else:
            poly_vtp = pp.merge_polys(poly_vtp, hold_vtp)
            skel_vtp = pp.merge_polys(skel_vtp, hold_skel_vtp)
        synth_tomo.add_network(
            net_helix, "Helix", entity_id, code=helix.get_type()
        )
        entity_id += 1

    # Loop for the list of input proteins loop
    count_prots = 0
    model_surfs, models, model_masks, model_codes = (
        list(),
        list(),
        list(),
        list(),
    )
    for p_id, p_file in tqdm(enumerate(PROTEINS_LIST),desc="Generating proteins"):

        print("\tPROCESSING FILE:", p_file)

        # Loading the protein
        protein = MmerFile(ROOT_PATH + "/" + p_file)

        # Generating the occupancy
        hold_occ = protein.get_pmer_occ()
        print(f"{p_id} protein_occupancy is {hold_occ}")
        if hasattr(hold_occ, "__len__"):
            hold_occ = OccGen(hold_occ).gen_occupancy()

        # Genrate the SAWLC network associated to the input protein
        # Polymer parameters
        # To read macromolecular models first we try to find the absolute path and secondly the relative to ROOT_PATH
        try:
            model = lio.load_mrc(protein.get_mmer_svol())
        except FileNotFoundError:
            model = lio.load_mrc(ROOT_PATH + "/" + protein.get_mmer_svol())
        # model = lio.load_mrc(ROOT_PATH + '/' + protein.get_mmer_svol())
        model = lin_map(model, lb=0, ub=1)
        model = vol_cube(model)
        model_mask = model < protein.get_iso()
        model[model_mask] = 0
        model_surf = pp.iso_surface(
            model, protein.get_iso(), closed=False, normals=None
        )
        if SURF_DEC is not None:
            model_surf = pp.poly_decimate(model_surf, SURF_DEC)
        center = 0.5 * np.asarray(model.shape, dtype=float)
        # Monomer centering
        model_surf = pp.poly_translate(model_surf, -center)
        # Voxel resolution scaling
        model_surf = pp.poly_scale(model_surf, VOI_VSIZE)
        model_surfs.append(model_surf)
        surf_diam = pp.poly_diam(model_surf) * protein.get_pmer_l()
        models.append(model)
        model_masks.append(model_mask)
        model_codes.append(protein.get_mmer_id())

        # Network generation
        pol_l_generator = PGenHelixFiber()
        if PROP_LIST is None:
            pol_s_generator = SGenUniform()
        else:
            assert len(PROP_LIST) == len(PROTEINS_LIST)
            pol_s_generator = SGenProp(PROP_LIST)
        net_sawlc = NetSAWLC(
            voi,
            VOI_VSIZE,
            protein.get_pmer_l() * surf_diam,
            model_surf,
            protein.get_pmer_l_max(),
            pol_l_generator,
            hold_occ,
            protein.get_pmer_over_tol(),
            poly=None,
            svol=model < protein.get_iso(),
            tries_mmer=MMER_TRIES,
            tries_pmer=PMER_TRIES,
        )
        # net_sawlc = NetSAWLCInter(voi, VOI_VSIZE, surf_diams, model_surfs, protein.get_pmer_l_max(),
        #                           pol_l_generator, pol_s_generator, protein.get_pmer_occ(), protein.get_pmer_over_tol(),
        #                           poly=None, svols=model_masks, codes=model_codes, compaq=5.5)
        net_sawlc.build_network()

        # Density tomogram updating
        net_sawlc.insert_density_svol(model_mask, voi, VOI_VSIZE, merge="min")
        net_sawlc.insert_density_svol(model, tomo_den, VOI_VSIZE, merge="max")
        hold_lbls = np.zeros(shape=tomo_lbls.shape, dtype=np.float32)
        net_sawlc.insert_density_svol(
            np.invert(model_mask), hold_lbls, VOI_VSIZE, merge="max"
        )
        tomo_lbls[hold_lbls > 0] = entity_id
        count_prots += net_sawlc.get_num_mmers()
        cp_voxels += (tomo_lbls == entity_id).sum()
        hold_vtp = net_sawlc.get_vtp()
        hold_skel_vtp = net_sawlc.get_skel()
        pp.add_label_to_poly(hold_vtp, entity_id, "Entity", mode="both")
        pp.add_label_to_poly(hold_skel_vtp, entity_id, "Entity", mode="both")
        pp.add_label_to_poly(hold_vtp, LBL_CP, "Type", mode="both")
        pp.add_label_to_poly(hold_skel_vtp, LBL_CP, "Type", mode="both")
        if poly_vtp is None:
            poly_vtp = hold_vtp
            skel_vtp = hold_skel_vtp
        else:
            poly_vtp = pp.merge_polys(poly_vtp, hold_vtp)
            skel_vtp = pp.merge_polys(skel_vtp, hold_skel_vtp)
        synth_tomo.add_network(
            net_sawlc, "SAWLC", entity_id, code=protein.get_mmer_id()
        )
        entity_id += 1

    # Loop for the list of input proteins loop
    count_mb_prots = 0
    if mbs_vtp is None:
        if len(MB_PROTEINS_LIST) > 0:
            print(
                "WARNING: membrane proteins can not inserted because there is no membrane surfaces!"
            )
    else:

        model_surfs, surf_diams, models, model_masks, model_codes = (
            list(),
            list(),
            list(),
            list(),
            list(),
        )
        for p_id, p_file in tqdm(enumerate(MB_PROTEINS_LIST),desc="Generating membrane proteins"):

            print("\tPROCESSING FILE:", p_file)

            # Loading the membrane protein
            protein = MmerMbFile(ROOT_PATH + "/" + p_file)

            # Generating the occupancy
            hold_occ = protein.get_pmer_occ()
            if hasattr(hold_occ, "__len__"):
                hold_occ = OccGen(hold_occ).gen_occupancy()

            # Insert membrane bound densities in a Polymer
            # Polymer parameters
            # To read macromolecular models first we try to find the absolute path and secondly the relative to ROOT_PATH
            try:
                model = lio.load_mrc(protein.get_mmer_svol())
            except FileNotFoundError:
                model = lio.load_mrc(ROOT_PATH + "/" + protein.get_mmer_svol())
            model = lin_map(model, lb=0, ub=1)
            model_mask = model < protein.get_iso()
            model[model_mask] = 0
            model_surf = iso_surface(
                model, protein.get_iso(), closed=False, normals=None
            )
            center = (
                protein.get_mmer_center()
            )  # .5 * np.asarray(model.shape, dtype=float)
            if center is None:
                center = 0.5 * (np.asarray(model.shape, dtype=float) - 1)
                off = np.asarray((0.0, 0.0, 0.0))
            else:
                center = np.asarray(center)
                off = 0.5 * np.asarray(model.shape) - center
            # Adding membrane domain to monomer surface
            mb_domain_mask = np.ones(shape=model.shape, dtype=bool)
            hold_mb_z_height = protein.get_mb_z_height()
            if hold_mb_z_height is None:
                hold_mb_z_height = int(round(center[2] + 2.5 / VOI_VSIZE))
            for z in range(hold_mb_z_height + 1, model.shape[2]):
                mb_domain_mask[:, :, z] = 0
            pp.add_sfield_to_poly(
                model_surf,
                mb_domain_mask,
                MB_DOMAIN_FIELD_STR,
                dtype="float",
                interp="NN",
                mode="points",
            )
            # Monomer centering
            model_surf = pp.poly_translate(model_surf, -center)
            # Voxel resolution scaling
            model_surf = pp.poly_scale(model_surf, VOI_VSIZE)
            surf_diam = pp.poly_diam(model_surf)
            pol_l_generator = PGenHelixFiber()
            # Network generation
            if protein.get_pmer_reverse_normals():
                mbs_vtp = pp.poly_reverse_normals(mbs_vtp)
            net_sawlc = NetSAWLC(
                voi,
                VOI_VSIZE,
                protein.get_pmer_l() * surf_diam,
                model_surf,
                protein.get_pmer_l_max(),
                pol_l_generator,
                hold_occ,
                protein.get_pmer_over_tol(),
                poly=mbs_vtp,
                svol=model < protein.get_iso(),
                tries_mmer=MMER_TRIES,
                tries_pmer=PMER_TRIES,
            )
            # net_sawlc = NetSAWLCInter(voi, VOI_VSIZE, protein.get_pmer_l() * surf_diam, model_surf, protein.get_pmer_l_max(),
            #                      pol_l_generator, protein.get_pmer_occ(), protein.get_pmer_over_tol(), poly=mb_poly,
            #                      svol=model < protein.get_iso())
            net_sawlc.build_network()
            # voi = net_sawlc.get_voi()

            # lio.write_mrc(voi.astype(np.float32), ROOT_PATH + '/hold_voi.mrc')
            # lio.write_mrc(set_mbs.get_tomo().astype(np.float32), ROOT_PATH + '/hold_den.mrc')

            # Density tomogram updating
            net_sawlc.insert_density_svol(
                model_mask, voi, VOI_VSIZE, merge="min"
            )
            net_sawlc.insert_density_svol(
                model, tomo_den, VOI_VSIZE, merge="max"
            )
            hold_lbls = np.zeros(shape=tomo_lbls.shape, dtype=np.float32)
            net_sawlc.insert_density_svol(
                np.invert(model_mask), hold_lbls, VOI_VSIZE, merge="max"
            )
            tomo_lbls[hold_lbls > 0] = entity_id
            count_mb_prots += net_sawlc.get_num_mmers()
            mp_voxels += (tomo_lbls == entity_id).sum()
            hold_vtp = net_sawlc.get_vtp()
            hold_skel_vtp = net_sawlc.get_skel()
            pp.add_label_to_poly(hold_vtp, entity_id, "Entity", mode="both")
            pp.add_label_to_poly(
                hold_skel_vtp, entity_id, "Entity", mode="both"
            )
            pp.add_label_to_poly(hold_vtp, LBL_MP, "Type", mode="both")
            pp.add_label_to_poly(hold_skel_vtp, LBL_MP, "Type", mode="both")
            if poly_vtp is None:
                poly_vtp = hold_vtp
                skel_vtp = hold_skel_vtp
            else:
                poly_vtp = pp.merge_polys(poly_vtp, hold_vtp)
                skel_vtp = pp.merge_polys(skel_vtp, hold_skel_vtp)
            synth_tomo.add_network(
                net_sawlc, "Mb-SAWLC", entity_id, code=protein.get_mmer_id()
            )
            entity_id += 1

    # Tomogram statistics
    print("\t\t-TOMOGRAM", str(tomod_id), "DENSITY STATISTICS:")
    print(
        "\t\t\t+Membranes:",
        count_mbs,
        "#, ",
        mb_voxels * vx_um3,
        "um**3, ",
        100.0 * (mb_voxels / voi_voxels),
        "%",
    )
    print(
        "\t\t\t+Actin:",
        count_actins,
        "#, ",
        ac_voxels * vx_um3,
        "um**3, ",
        100.0 * (ac_voxels / voi_voxels),
        "%",
    )
    print(
        "\t\t\t+Microtublues:",
        count_mts,
        "#, ",
        mt_voxels * vx_um3,
        "um**3, ",
        100.0 * (mt_voxels / voi_voxels),
        "%",
    )
    print(
        "\t\t\t+Proteins:",
        count_prots,
        "#, ",
        cp_voxels * vx_um3,
        "um**3, ",
        100.0 * (cp_voxels / voi_voxels),
        "%",
    )
    print(
        "\t\t\t+Membrane proteins:",
        count_mb_prots,
        "#, ",
        mp_voxels * vx_um3,
        "um**3, ",
        100.0 * (mp_voxels / voi_voxels),
        "%",
    )
    counts_total = (
        count_mbs + count_actins + count_mts + count_prots + count_mb_prots
    )
    total_voxels = mb_voxels + ac_voxels + mt_voxels + cp_voxels + mp_voxels
    print(
        "\t\t\t+Total:",
        counts_total,
        "#, ",
        total_voxels * vx_um3,
        "um**3, ",
        100.0 * (total_voxels / voi_voxels),
        "%",
    )
    print(
        "\t\t\t+Time for generation: ", (time.time() - hold_time) / 60, "mins"
    )

    # Storing simulated density results
    tomo_den_out = TOMOS_DIR + "/tomo_den_" + str(tomod_id) + ".mrc"
    lio.write_mrc(tomo_den, tomo_den_out, v_size=VOI_VSIZE)
    synth_tomo.set_den(tomo_den_out)
    tomo_lbls_out = TOMOS_DIR + "/tomo_lbls_" + str(tomod_id) + ".mrc"
    lio.write_mrc(tomo_lbls, tomo_lbls_out, v_size=VOI_VSIZE)
    poly_den_out = TOMOS_DIR + "/poly_den_" + str(tomod_id) + ".vtp"
    lio.save_vtp(poly_vtp, poly_den_out)
    synth_tomo.set_poly(poly_den_out)
    poly_skel_out = TOMOS_DIR + "/poly_skel_" + str(tomod_id) + ".vtp"
    lio.save_vtp(skel_vtp, poly_skel_out)

    """# TEM for 3D reconstructions
    temic = tem.TEM(TEM_DIR)
    vol = lio.load_mrc(tomo_den_out)
    temic.gen_tilt_series_imod(vol, TILT_ANGS, ax="Y")
    temic.add_mics_misalignment(MALIGN_MN, MALIGN_MX, MALIGN_SG)
    if DETECTOR_SNR is not None:
        if hasattr(DETECTOR_SNR, "__len__"):
            if len(DETECTOR_SNR) >= 2:
                snr = round(
                    (DETECTOR_SNR[1] - DETECTOR_SNR[0]) * random.random()
                    + DETECTOR_SNR[0],
                    2,
                )
            else:
                snr = DETECTOR_SNR[0]
        else:
            snr = DETECTOR_SNR
        
        temic.add_detector_noise(snr)
    temic.invert_mics_den()
    temic.set_header(data="mics", p_size=(VOI_VSIZE, VOI_VSIZE, VOI_VSIZE))
    temic.recon3D_imod()
    temic.set_header(
        data="rec3d", p_size=(VOI_VSIZE, VOI_VSIZE, VOI_VSIZE), origin=(0, 0, 0)
    )
    if DETECTOR_SNR is not None:
        out_mics, out_tomo_rec = (
            TOMOS_DIR
            + "/tomo_mics_"
            + str(tomod_id)
            + "_snr"
            + str(snr)
            + ".mrc",
            TOMOS_DIR
            + "/tomo_rec_"
            + str(tomod_id)
            + "_snr"
            + str(snr)
            + ".mrc",
        )
    else:
        out_mics, out_tomo_rec = (
            TOMOS_DIR + "/tomo_mics_" + str(tomod_id) + ".mrc",
            TOMOS_DIR + "/tomo_rec_" + str(tomod_id) + ".mrc",
        )
    shutil.copyfile(TEM_DIR + "/out_micrographs.mrc", out_mics)
    shutil.copyfile(TEM_DIR + "/out_rec3d.mrc", out_tomo_rec)
    synth_tomo.set_mics(out_mics)
    synth_tomo.set_tomo(out_tomo_rec)"""

    # Update the set
    set_stomos.add_tomos(synth_tomo)


##Storing tomograms CSV file
set_stomos.save_csv(OUT_DIR + "/tomos_motif_list.csv")
# Path to the CSV file
csv_file_path = os.path.join(OUT_DIR, "tomos_motif_list.csv")

# Display statistics from the CSV file
display_statistics_from_csv(csv_file_path)

print("Successfully terminated. (" + time.strftime("%c") + ")")
