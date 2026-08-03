'''
Program to collate Boltz-2 results: parses confidence/affinity JSON output, computes
center-of-mass-based distance metrics (distance to ligand, distance to the orthosteric
binding site) and protein-ligand hydrogen bond counts, and exports a summary CSV.

Usage:
python collate_boltz2.py --binding-site-residue1 157 --output-prefix boltz_results

Sample config.json:

{
    "binding_site_residue1": 120,
    "binding_site_residue2": 150,
    "binding_site_residue3": null,
    "output_prefix": "boltz_results"
}

The script needs to be run from the Boltz-2 base directory (the directory containing the
`boltz_results_*` and `yaml*` folders) with the appropriate anaconda environment activated.

Outputs:
    <output_prefix>_full.csv  - all computed columns (confidence/affinity metrics, SMILES,
                                 distance_to_ligand_com, distance_to_orthosteric_site, num_hbonds)
    <output_prefix>.csv       - chembl_id, affinity_pred_value, affinity_probability_binary,
                                 confidence_score only

Checkpointing (for long batch runs over many files, e.g. on an HPC cluster):
    While the hydrogen-bond and distance calculations run, each model's result is written
    to a checkpoint CSV as soon as it completes (not just at the end of the run):
        <output_prefix>_hbonds_checkpoint.csv
        <output_prefix>_distances_checkpoint_<site1>_<site2>_<site3>.csv
    If the job is killed or crashes partway through (e.g. a SLURM walltime limit), simply
    re-running the same command resumes from these checkpoints instead of starting over.
    They are deleted automatically once a run completes successfully. Note the distance
    checkpoint is keyed by the binding-site residue configuration, since
    distance_to_orthosteric_site depends on it -- changing --binding-site-residue1/2/3
    between runs (with the same --output-prefix) starts a fresh checkpoint rather than
    silently reusing distances computed for a different site.

'''

# Imports

import os
import re
import csv
import pandas as pd
import json
import numpy as np
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path
from typing import List, Optional
import logging
import biotite.structure as structure
import biotite.structure.io.pdb as pdb
import argparse
import yaml


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Columns required in the final, strict-schema output CSV.
FINAL_CSV_COLUMNS = ['chembl_id', 'affinity_pred_value', 'affinity_probability_binary', 'confidence_score']

CHEMBL_ID_PATTERN = re.compile(r'CHEMBL\d+')

# Functions________________________________________


def parse_arguments():
    """Parse command line arguments for configuration options."""
    parser = argparse.ArgumentParser(
        description="Collate Boltz-2 results: parse confidence/affinity data, compute "
                     "ligand distance and hydrogen bond metrics, and export a summary CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--binding-site-residue1',
        type=int,
        default=157,
        help='Primary binding site (orthosteric site) residue number'
    )

    parser.add_argument(
        '--binding-site-residue2',
        type=int,
        default=None,
        help='Secondary binding site residue number (optional)'
    )

    parser.add_argument(
        '--binding-site-residue3',
        type=int,
        default=None,
        help='Tertiary binding site residue number (optional)'
    )

    parser.add_argument(
        '--output-prefix',
        type=str,
        default='boltz_results',
        help='Prefix for output CSV files'
    )

    parser.add_argument(
        '--config-file',
        type=str,
        help='JSON configuration file (overrides command line arguments)'
    )

    return parser.parse_args()


def load_config_from_file(config_file):
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_file}: {e}")
        return {}


def get_smiles_from_yaml(base_name):
    """Search yaml directories for the YAML file and extract the SMILES string."""
    yaml_dirs = [f"yaml{i}" for i in range(1, 9)] + ["yaml"]
    for yaml_dir in yaml_dirs:
        yaml_file = os.path.join(yaml_dir, f"{base_name}.yaml")
        if os.path.exists(yaml_file):
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                    # Find the ligand entry in sequences
                    for entry in data.get('sequences', []):
                        ligand = entry.get('ligand')
                        if ligand and 'smiles' in ligand:
                            return ligand['smiles']
            except Exception as e:
                print(f"Error reading {yaml_file}: {e}")
                return None
    return None


def extract_chembl_id(base_name: str) -> str:
    """
    Extract a ChEMBL ID (e.g. 'CHEMBL3039503') out of a base_name string such as
    '1036112_CHEMBL3039503'. Falls back to the raw base_name (and logs a warning) if no
    CHEMBL\\d+ substring is found, so a row is never silently dropped/blanked just because
    it doesn't follow the usual naming convention.
    """
    match = CHEMBL_ID_PATTERN.search(base_name)
    if match:
        return match.group(0)
    logger.warning(f"No ChEMBL ID pattern found in base_name '{base_name}'; using raw base_name as fallback")
    return base_name


def parse_boltz2_results(directory):
    results = []

    # Get confidence and affinity files
    confidence_files = [f for f in os.listdir(directory) if f.startswith('confidence_') and f.endswith('.json')]
    affinity_files = [f for f in os.listdir(directory) if f.startswith('affinity_') and f.endswith('.json')]

    # Create a dictionary to store affinity data by model key
    affinity_dict = {}

    # Process affinity files first to build lookup dictionary
    for aff_file in affinity_files:
        base_name = aff_file.replace('affinity_', '').replace('.json', '')  # Extract base name
        model_key = f"{base_name}"  # Key matches confidence file base name without `_model_0`

        try:
            affinity_data = json.load(open(os.path.join(directory, aff_file), 'r'))
            affinity_dict[model_key] = affinity_data
        except Exception as e:
            print(f"Error loading affinity file {aff_file}: {e}")

    # Process confidence files and merge with affinity data
    for conf_file in confidence_files:
        model_index = conf_file.split('_')[-1].split('.')[0]  # Extract model index from filename
        base_name = conf_file.replace('confidence_', '').replace(f'_model_{model_index}.json', '')  # Extract base name
        model_file = f"{base_name}_model_{model_index}.pdb"  # Assuming the model files are in PDB format
        model_path = os.path.join(directory, model_file)
        model_key = f"{base_name}"  # Key matches affinity file base name

        try:
            confidence_data = json.load(open(os.path.join(directory, conf_file), 'r'))
        except Exception as e:
            print(f"Error loading confidence file {conf_file}: {e}")
            continue

        if os.path.exists(model_path):
            result_entry = {
                'model_path': model_path,
                'model_index': model_index,
                'chembl_id': extract_chembl_id(base_name),
                'confidence_score': confidence_data['confidence_score'],
                'ptm': confidence_data['ptm'],
                'iptm': confidence_data['iptm'],
                'ligand_iptm': confidence_data['ligand_iptm'],
                'protein_iptm': confidence_data['protein_iptm'],
                'complex_plddt': confidence_data['complex_plddt'],
                'complex_iplddt': confidence_data['complex_iplddt'],
                'complex_pde': confidence_data['complex_pde'],
                'complex_ipde': confidence_data['complex_ipde'],
                'chains_ptm': confidence_data['chains_ptm'],
                'pair_chains_iptm': confidence_data['pair_chains_iptm']
            }

            # Add affinity data if available for this model
            if model_key in affinity_dict:
                affinity_data = affinity_dict[model_key]
                result_entry.update({
                    'affinity_pred_value': affinity_data.get('affinity_pred_value', None),
                    'affinity_probability_binary': affinity_data.get('affinity_probability_binary', None),
                    'affinity_pred_value1': affinity_data.get('affinity_pred_value1', None),
                    'affinity_probability_binary1': affinity_data.get('affinity_probability_binary1', None),
                    'affinity_pred_value2': affinity_data.get('affinity_pred_value2', None),
                    'affinity_probability_binary2': affinity_data.get('affinity_probability_binary2', None)
                })
            else:
                result_entry.update({
                    'affinity_pred_value': None,
                    'affinity_probability_binary': None,
                    'affinity_pred_value1': None,
                    'affinity_probability_binary1': None,
                    'affinity_pred_value2': None,
                    'affinity_probability_binary2': None
                })

            # Add SMILES string from YAML (search all yaml dirs)
            result_entry['smiles'] = get_smiles_from_yaml(base_name)

            results.append(result_entry)
        else:
            print(f"Model file does not exist: {model_path}")

    print(f"Affinity data matched for {sum(1 for r in results if r.get('affinity_pred_value') is not None)} models")

    return pd.DataFrame(results)


# Function to combine the CSV results from multiple files:
def combine_csv_results(file_list):
    combined_df = pd.concat([pd.read_csv(f) for f in file_list if f.endswith('_results.csv')])
    return combined_df


# Function to average the distance to up to 3 configured binding-site (orthosteric site) residues.
def calculate_orthosteric_distance(distances, site1, site2=None, site3=None) -> Optional[float]:
    """
    Average the CA-to-ligand-COM distance at up to 3 configured binding-site residue
    positions (renamed/repurposed from the original threshold-filtering version of this
    function — now returns the averaged distance value instead of a pass/fail bool).

    Parameters:
    distances (np.array): per-residue CA-to-ligand-COM distances, indexed by residue position
    site1, site2, site3 (int): residue position(s) to average over (site2/site3 optional)

    Returns: the averaged distance (float), or None if distances is empty/None, site1 is
    None, or a configured site index is out of range for the array.
    """
    if distances is None or len(distances) == 0 or site1 is None:
        return None

    sites = [s for s in (site1, site2, site3) if s is not None]

    try:
        site_distances = np.array([distances[s] for s in sites])
    except IndexError:
        logger.warning(f"Binding site residue index out of range for distances array of length {len(distances)}")
        return None

    return float(np.mean(site_distances))


# Function to extract CA positions from a specified chain in the PDB data
def extract_ca_positions(pdb_data, chain_id):
    """
    Extract CA positions from a specified chain in the PDB data.

    Parameters:
    -----------
    pdb_data : AtomArray or AtomArrayStack
        The PDB structure data.
    chain_id : str
        The chain ID to extract CA positions from.

    Returns:
    --------
    AtomArray
        The CA atoms from the specified chain.
    """
    if isinstance(pdb_data, structure.AtomArrayStack):
        pdb_data = pdb_data[0]  # Use the first model if it's an AtomArrayStack
    chain_atoms = pdb_data[pdb_data.chain_id == chain_id]
    ca_atoms = chain_atoms[chain_atoms.atom_name == 'CA']
    return ca_atoms

# Function to extract all atoms from a specified chain in the PDB data
def extract_atoms_by_chain(pdb_data, chain_id):
    """
    Extract all atoms from a specified chain in the PDB data.

    Parameters:
    -----------
    pdb_data : AtomArray or AtomArrayStack
        The PDB structure data.
    chain_id : str
        The chain ID to extract atoms from.

    Returns:
    --------
    AtomArray
        All atoms from the specified chain.
    """
    if isinstance(pdb_data, structure.AtomArrayStack):
        pdb_data = pdb_data[0]  # Use the first model if it's an AtomArrayStack
    chain_atoms = pdb_data[pdb_data.chain_id == chain_id]
    return chain_atoms


# Utility Functions for calculating Centre of Mass and distances.
def calculate_center_of_mass(atoms):
    """Calculate the center of mass of a given set of atoms."""
    if len(atoms) == 0:
        return None
    positions = atoms.coord  # Use .coord instead of .get_positions()
    return np.mean(positions, axis=0)

# Calculate the distances between the CA atoms in chain A and the center of mass of chain B
def calculate_distances_to_com(ca_atoms, com):
    """Calculate distances from CA atoms to a given center of mass."""
    if com is None or len(ca_atoms) == 0:
        return None
    ca_positions = ca_atoms.coord  # Use .coord instead of .get_positions()
    distances = np.linalg.norm(ca_positions - com, axis=1)
    return distances


# Hydrogen Bond Calculation Functions
def calculate_hydrogen_bonds(chain_a_atoms, chain_b_atoms, distance_cutoff=3.5, angle_cutoff=120):
    """
    Calculate potential hydrogen bonds between Chain A and Chain B based on heavy atom distances
    and estimated angles. Since explicit hydrogens are not present, we approximate geometry using
    bonded atoms.

    Parameters:
    -----------
    chain_a_atoms : AtomArray
        Atoms from Chain A (protein)
    chain_b_atoms : AtomArray
        Atoms from Chain B (ligand)
    distance_cutoff : float, default=3.5
        Maximum distance (in Angstroms) between donor and acceptor heavy atoms
    angle_cutoff : float, default=120
        Minimum angle (in degrees) for hydrogen bond geometry (approximated)

    Returns:
    --------
    tuple: (hbonds, bond_info)
        hbonds: list of potential hydrogen bonds
        bond_info: detailed information about each bond
    """

    # Define potential hydrogen bond donors and acceptors (heavy atoms only)
    donor_atoms = ['N', 'O']  # Atoms that can donate hydrogen
    acceptor_atoms = ['N', 'O', 'S', 'F']  # Atoms that can accept hydrogen

    # Get donor and acceptor atoms from each chain
    chain_a_donors = chain_a_atoms[np.isin(chain_a_atoms.element, donor_atoms)]
    chain_a_acceptors = chain_a_atoms[np.isin(chain_a_atoms.element, acceptor_atoms)]

    chain_b_donors = chain_b_atoms[np.isin(chain_b_atoms.element, donor_atoms)]
    chain_b_acceptors = chain_b_atoms[np.isin(chain_b_atoms.element, acceptor_atoms)]

    def find_bonded_atom(target_atom, atom_array, bond_distance=1.8):
        """Find a bonded atom to estimate hydrogen position"""
        distances = np.linalg.norm(atom_array.coord - target_atom.coord, axis=1)
        bonded_indices = np.where((distances > 0) & (distances <= bond_distance))[0]
        if len(bonded_indices) > 0:
            return atom_array[bonded_indices[0]]  # Return first bonded atom
        return None

    def calculate_angle(atom1_coord, atom2_coord, atom3_coord):
        """Calculate angle between three points (atom2 is the vertex)"""
        vec1 = atom1_coord - atom2_coord
        vec2 = atom3_coord - atom2_coord

        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Calculate angle
        cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    hydrogen_bonds = []
    bond_info = []

    # Check Chain A donors to Chain B acceptors
    for donor in chain_a_donors:
        for acceptor in chain_b_acceptors:
            distance = np.linalg.norm(donor.coord - acceptor.coord)

            if distance <= distance_cutoff:
                # Find a bonded atom to donor to estimate hydrogen position
                bonded_to_donor = find_bonded_atom(donor, chain_a_atoms)

                if bonded_to_donor is not None:
                    # Calculate angle: bonded_atom - donor - acceptor
                    angle = calculate_angle(bonded_to_donor.coord, donor.coord, acceptor.coord)

                    if angle >= angle_cutoff:
                        hydrogen_bonds.append((donor, acceptor))
                        bond_info.append({
                            'donor_atom': donor.atom_name,
                            'donor_res': donor.res_name,
                            'donor_res_id': donor.res_id,
                            'donor_chain': donor.chain_id,
                            'acceptor_atom': acceptor.atom_name,
                            'acceptor_res': acceptor.res_name,
                            'acceptor_res_id': acceptor.res_id,
                            'acceptor_chain': acceptor.chain_id,
                            'distance': distance,
                            'angle': angle,
                            'donor_coord': donor.coord,
                            'acceptor_coord': acceptor.coord
                        })
                else:
                    # If no bonded atom found, accept based on distance only
                    hydrogen_bonds.append((donor, acceptor))
                    bond_info.append({
                        'donor_atom': donor.atom_name,
                        'donor_res': donor.res_name,
                        'donor_res_id': donor.res_id,
                        'donor_chain': donor.chain_id,
                        'acceptor_atom': acceptor.atom_name,
                        'acceptor_res': acceptor.res_name,
                        'acceptor_res_id': acceptor.res_id,
                        'acceptor_chain': acceptor.chain_id,
                        'distance': distance,
                        'angle': None,
                        'donor_coord': donor.coord,
                        'acceptor_coord': acceptor.coord
                    })

    # Check Chain B donors to Chain A acceptors
    for donor in chain_b_donors:
        for acceptor in chain_a_acceptors:
            distance = np.linalg.norm(donor.coord - acceptor.coord)

            if distance <= distance_cutoff:
                bonded_to_donor = find_bonded_atom(donor, chain_b_atoms)

                if bonded_to_donor is not None:
                    angle = calculate_angle(bonded_to_donor.coord, donor.coord, acceptor.coord)

                    if angle >= angle_cutoff:
                        hydrogen_bonds.append((donor, acceptor))
                        bond_info.append({
                            'donor_atom': donor.atom_name,
                            'donor_res': donor.res_name,
                            'donor_res_id': donor.res_id,
                            'donor_chain': donor.chain_id,
                            'acceptor_atom': acceptor.atom_name,
                            'acceptor_res': acceptor.res_name,
                            'acceptor_res_id': acceptor.res_id,
                            'acceptor_chain': acceptor.chain_id,
                            'distance': distance,
                            'angle': angle,
                            'donor_coord': donor.coord,
                            'acceptor_coord': acceptor.coord
                        })
                else:
                    hydrogen_bonds.append((donor, acceptor))
                    bond_info.append({
                        'donor_atom': donor.atom_name,
                        'donor_res': donor.res_name,
                        'donor_res_id': donor.res_id,
                        'donor_chain': donor.chain_id,
                        'acceptor_atom': acceptor.atom_name,
                        'acceptor_res': acceptor.res_name,
                        'acceptor_res_id': acceptor.res_id,
                        'acceptor_chain': acceptor.chain_id,
                        'distance': distance,
                        'angle': None,
                        'donor_coord': donor.coord,
                        'acceptor_coord': acceptor.coord
                    })

    return hydrogen_bonds, bond_info

def analyze_hydrogen_bonds(chain_a_atoms, chain_b_atoms, distance_cutoff=3.5):
    """
    Analyze and display hydrogen bond information between two chains using heavy atoms only.
    """
    hbonds, bond_details = calculate_hydrogen_bonds(
        chain_a_atoms, chain_b_atoms, distance_cutoff
    )

    print(f"\nFound {len(hbonds)} potential hydrogen bonds (heavy atom distance < {distance_cutoff} Å)")
    return len(hbonds), bond_details

def process_hydrogen_bonds(file_path):
    """Process a single PDB file for hydrogen bond analysis. Module-level,
    multiprocessing-picklable worker; returns a dict so results can be streamed to a
    checkpoint file as they complete (see BoltzCollator.calculate_hydrogen_bonds)."""
    if not os.path.isfile(file_path):
        return None

    try:
        pdb_file_obj = pdb.PDBFile.read(file_path)
        pdb_data = pdb_file_obj.get_structure()

        # Select atoms from Chain A and Chain B
        chain_a_atoms = extract_atoms_by_chain(pdb_data, chain_id='A')
        chain_b_atoms = extract_atoms_by_chain(pdb_data, chain_id='B')

        # Analyze hydrogen bonds
        num_hbonds, hbond_details = analyze_hydrogen_bonds(chain_a_atoms, chain_b_atoms, distance_cutoff=3.5)

        return {'model_path': file_path, 'num_hbonds': num_hbonds}
    except Exception as e:
        # Broad except: in a long unattended batch run over thousands of files, one
        # malformed/corrupt PDB should not take down the whole pool.
        print(f"Error processing file {file_path}: {e}")
        return None


def process_model_geometry(file_path, site1=None, site2=None, site3=None):
    """
    Module-level, multiprocessing-picklable worker that computes, for one PDB model:
      - distance_to_ligand_com: distance between the protein's (chain A CA atoms) center of
        mass and the ligand's (chain B) center of mass
      - distance_to_orthosteric_site: averaged CA-to-ligand-COM distance at the configured
        binding site residue position(s)
    """
    if not os.path.isfile(file_path):
        return None

    try:
        pdb_file_obj = pdb.PDBFile.read(file_path)
        pdb_data = pdb_file_obj.get_structure()

        ca_atoms = extract_ca_positions(pdb_data, chain_id='A')
        ligand_atoms = extract_atoms_by_chain(pdb_data, chain_id='B')

        protein_com = calculate_center_of_mass(ca_atoms)
        ligand_com = calculate_center_of_mass(ligand_atoms)

        distance_to_ligand_com = None
        if protein_com is not None and ligand_com is not None:
            distance_to_ligand_com = float(np.linalg.norm(protein_com - ligand_com))

        distance_to_orthosteric_site = None
        if ligand_com is not None:
            ca_to_ligand_distances = calculate_distances_to_com(ca_atoms, ligand_com)
            distance_to_orthosteric_site = calculate_orthosteric_distance(
                ca_to_ligand_distances, site1, site2, site3
            )

        return {
            'model_path': file_path,
            'distance_to_ligand_com': distance_to_ligand_com,
            'distance_to_orthosteric_site': distance_to_orthosteric_site,
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def _hbonds_worker(args):
    """Adapter so process_hydrogen_bonds can be used with pool.imap_unordered, which
    passes a single argument per call (needed for streaming/checkpointed processing)."""
    (file_path,) = args
    return process_hydrogen_bonds(file_path)


def _geometry_worker(args):
    """Adapter so process_model_geometry can be used with pool.imap_unordered."""
    file_path, site1, site2, site3 = args
    return process_model_geometry(file_path, site1, site2, site3)


# End of Functions________________________________________


class BoltzCollator:
    """Main class for collating Boltz-2 results: parses confidence/affinity data, computes
    ligand distance and hydrogen bond metrics, and exports the summary CSVs."""

    def __init__(self, config: dict):
        self.config = config
        self.base_dir = Path(config.get('working_directory', '.'))
        self.results_df = None
        self.hydrogen_bond_df = None
        self.distance_df = None
        self.pdb_file_list = []

    def get_boltz_folders(self) -> List[str]:
        """Get list of Boltz result folders."""
        return [f for f in os.listdir(self.base_dir) if f.startswith('boltz_results_')]

    def _normalize_path(self, path: str) -> str:
        """Normalize file paths for consistent merging."""
        path_obj = Path(path)

        if path_obj.is_absolute():
            try:
                path_obj = path_obj.relative_to(Path.cwd())
            except ValueError:
                pass

        normalized = str(path_obj)
        if not normalized.startswith('./'):
            normalized = f"./{normalized}"

        return normalized

    def parse_boltz_results(self) -> pd.DataFrame:
        """Parse Boltz-2 results from all folders."""
        logger.info("Parsing Boltz-2 results...")
        folder_list = self.get_boltz_folders()
        csv_files = []

        for folder in folder_list:
            # Dynamically determine the correct subfolder name
            folder_id = folder.split('_', 2)[2]  # Extract the folder ID (e.g., "1036112_CHEMBL")
            predictions_folder = self.base_dir / folder / 'predictions' / folder_id

            if not predictions_folder.exists():
                logger.error(f"Predictions folder does not exist: {predictions_folder}")
                continue

            logger.info(f"Processing folder: {predictions_folder}")
            results_df = parse_boltz2_results(str(predictions_folder))
            csv_filename = f"{folder}_results.csv"
            results_df.to_csv(csv_filename, index=False)
            csv_files.append(csv_filename)

        # Combine all CSV files
        self.results_df = combine_csv_results(csv_files)

        if 'model_path' in self.results_df.columns:
            self.results_df['model_path'] = self.results_df['model_path'].apply(self._normalize_path)
        else:
            logger.error("No 'model_path' column found in parsed results")

        self._cleanup_temp_files(csv_files)

        return self.results_df

    def _cleanup_temp_files(self, file_list: List[str]) -> None:
        """Remove temporary CSV files."""
        for file in file_list:
            if Path(file).exists():
                os.remove(file)

    def collect_pdb_files(self) -> List[str]:
        """Collect all PDB file paths from both possible prediction subfolders."""
        self.pdb_file_list = []
        folder_list = self.get_boltz_folders()

        for folder in folder_list:
            # Extract folder ID and full folder name
            folder_id = folder.split('_', 2)[2]
            predictions_folder1 = self.base_dir / folder / 'predictions' / folder_id
            predictions_folder2 = self.base_dir / folder / 'predictions' / folder  # Use full folder name

            # Check both possible prediction folders
            for predictions_folder in [predictions_folder1, predictions_folder2]:
                if predictions_folder.exists():
                    pdb_files = list(predictions_folder.glob('*.pdb'))
                    self.pdb_file_list.extend([str(f) for f in pdb_files])

        logger.info(f"Found {len(self.pdb_file_list)} PDB files to process")
        return self.pdb_file_list

    def _run_checkpointed_pool(self, checkpoint_path: str, fieldnames: List[str], worker, work_items) -> pd.DataFrame:
        """
        Run `worker` over `work_items` (an iterable of arg-tuples, first element of each
        tuple must be the model_path) via a multiprocessing pool, using imap_unordered so
        each completed result is written to `checkpoint_path` (and flushed to disk)
        immediately rather than only at the end. This means a crash, an HPC walltime
        kill, or any other mid-run failure loses at most the in-flight batch, not the
        whole run -- simply re-running picks up from the checkpoint automatically.

        Returns a DataFrame (columns = fieldnames, 'model_path' normalized) restricted to
        the model paths present in `work_items`, so stale rows from an unrelated older
        checkpoint at the same path are ignored.
        """
        work_items = list(work_items)
        all_paths = [item[0] for item in work_items]

        done_paths = set()
        if os.path.exists(checkpoint_path):
            existing = pd.read_csv(checkpoint_path)
            done_paths = set(existing['model_path']) & set(all_paths)
            if done_paths:
                logger.info(
                    f"Resuming from checkpoint '{checkpoint_path}': "
                    f"{len(done_paths)}/{len(all_paths)} models already done"
                )

        remaining_items = [item for item in work_items if item[0] not in done_paths]

        if remaining_items:
            start_time = time.time()
            num_processes = max(1, min(mp.cpu_count(), len(remaining_items)))
            write_header = not os.path.exists(checkpoint_path)

            with open(checkpoint_path, 'a', newline='') as ckpt_file:
                writer = csv.writer(ckpt_file)
                if write_header:
                    writer.writerow(fieldnames)
                with mp.Pool(processes=num_processes) as pool:
                    for result in pool.imap_unordered(worker, remaining_items):
                        if result is not None:
                            writer.writerow([result[field] for field in fieldnames])
                            ckpt_file.flush()

            logger.info(
                f"Processed {len(remaining_items)} new models in "
                f"{time.time() - start_time:.2f} seconds"
            )
        else:
            logger.info("All models already checkpointed, nothing new to compute")

        if not os.path.exists(checkpoint_path):
            return pd.DataFrame(columns=fieldnames)

        checkpoint_df = pd.read_csv(checkpoint_path)
        result_df = checkpoint_df[checkpoint_df['model_path'].isin(all_paths)].drop_duplicates(
            subset='model_path', keep='last'
        ).copy()
        result_df['model_path'] = result_df['model_path'].apply(self._normalize_path)
        return result_df

    def calculate_hydrogen_bonds(self) -> pd.DataFrame:
        """Calculate hydrogen bonds using parallel, checkpointed processing."""
        logger.info("Starting hydrogen bond analysis...")

        if not self.pdb_file_list:
            self.collect_pdb_files()

        if not self.pdb_file_list:
            logger.error("No PDB files found for hydrogen bond analysis.")
            self.hydrogen_bond_df = pd.DataFrame()
            return self.hydrogen_bond_df

        prefix = self.config.get('output_prefix', 'boltz_results')
        checkpoint_path = f"{prefix}_hbonds_checkpoint.csv"

        hydrogen_bond_df = self._run_checkpointed_pool(
            checkpoint_path,
            ['model_path', 'num_hbonds'],
            _hbonds_worker,
            [(f,) for f in self.pdb_file_list],
        )
        hydrogen_bond_df = hydrogen_bond_df.set_index('model_path')

        self.hydrogen_bond_df = hydrogen_bond_df
        return self.hydrogen_bond_df

    def calculate_distance_metrics(self) -> pd.DataFrame:
        """Calculate distance-to-ligand-COM and distance-to-orthosteric-site using
        parallel, checkpointed processing."""
        logger.info("Calculating distance metrics...")

        if not self.pdb_file_list:
            self.collect_pdb_files()

        if not self.pdb_file_list:
            logger.error("No PDB files found for distance calculation.")
            self.distance_df = pd.DataFrame()
            return self.distance_df

        site1 = self.config.get('binding_site_residue1')
        site2 = self.config.get('binding_site_residue2')
        site3 = self.config.get('binding_site_residue3')

        # Keyed by the binding-site config: distance_to_orthosteric_site depends on it, so
        # changing residues between runs (with the same --output-prefix) must not silently
        # reuse distances computed for a different site.
        site_key = f"{site1}_{site2 if site2 is not None else 'none'}_{site3 if site3 is not None else 'none'}"
        prefix = self.config.get('output_prefix', 'boltz_results')
        checkpoint_path = f"{prefix}_distances_checkpoint_{site_key}.csv"

        distance_df = self._run_checkpointed_pool(
            checkpoint_path,
            ['model_path', 'distance_to_ligand_com', 'distance_to_orthosteric_site'],
            _geometry_worker,
            [(f, site1, site2, site3) for f in self.pdb_file_list],
        )

        self.distance_df = distance_df
        return self.distance_df

    def merge_all_data(self) -> pd.DataFrame:
        """Merge hydrogen bond and distance metrics into the parsed results."""
        logger.info("Merging all data...")

        if self.results_df is None or self.results_df.empty:
            logger.error("Results DataFrame is None or empty. Run parse_boltz_results() first.")
            return pd.DataFrame()

        if self.hydrogen_bond_df is not None and not self.hydrogen_bond_df.empty:
            self.results_df = pd.merge(
                self.results_df,
                self.hydrogen_bond_df,
                left_on='model_path',
                right_index=True,
                how='left'
            )
        else:
            logger.warning("No hydrogen bond data to merge")

        if self.distance_df is not None and not self.distance_df.empty:
            self.results_df = pd.merge(
                self.results_df,
                self.distance_df,
                on='model_path',
                how='left'
            )
        else:
            logger.warning("No distance data to merge")

        logger.info(f"Merged results shape: {self.results_df.shape}")
        return self.results_df

    def export_full_csv(self, output_path: str) -> None:
        """Export the full merged DataFrame with all computed columns."""
        if self.results_df is None or self.results_df.empty:
            logger.error("No results to export.")
            return
        self.results_df.to_csv(output_path, index=False)
        logger.info(f"Full results exported to '{output_path}' ({len(self.results_df)} rows)")

    def export_final_csv(self, output_path: str) -> None:
        """Export exactly the required columns: chembl_id, affinity_pred_value,
        affinity_probability_binary, confidence_score."""
        if self.results_df is None or self.results_df.empty:
            logger.error("No results to export.")
            return

        missing = [c for c in FINAL_CSV_COLUMNS if c not in self.results_df.columns]
        if missing:
            logger.error(f"Missing required columns for final CSV: {missing}")
            return

        self.results_df[FINAL_CSV_COLUMNS].to_csv(output_path, index=False)
        logger.info(f"Final results exported to '{output_path}' ({len(self.results_df)} rows)")

    def run_collation(self) -> pd.DataFrame:
        """Run the full collation pipeline."""
        logger.info("Starting Boltz-2 collation pipeline...")

        self.parse_boltz_results()
        if self.results_df is None or self.results_df.empty:
            raise ValueError("Failed to parse Boltz results - no data found")

        logger.info(f"Parsed results shape: {self.results_df.shape}")

        self.calculate_hydrogen_bonds()
        self.calculate_distance_metrics()
        self.merge_all_data()

        prefix = self.config.get('output_prefix', 'boltz_results')
        self.export_full_csv(f"{prefix}_full.csv")
        self.export_final_csv(f"{prefix}.csv")

        self._cleanup_checkpoints()

        logger.info("Collation pipeline completed successfully!")
        logger.info(f"Final results: {len(self.results_df)} models collated")

        return self.results_df

    def _cleanup_checkpoints(self) -> None:
        """Remove the resume checkpoint files now that the run finished successfully.
        They're only needed to survive a mid-run crash/kill; keeping them around after a
        clean completion just risks a future run silently resuming from stale data."""
        prefix = self.config.get('output_prefix', 'boltz_results')
        for checkpoint_file in Path('.').glob(f"{prefix}_hbonds_checkpoint.csv"):
            checkpoint_file.unlink()
        for checkpoint_file in Path('.').glob(f"{prefix}_distances_checkpoint_*.csv"):
            checkpoint_file.unlink()


def main():
    """Main execution function."""
    args = parse_arguments()

    config = {
        'binding_site_residue1': args.binding_site_residue1,
        'binding_site_residue2': args.binding_site_residue2,
        'binding_site_residue3': args.binding_site_residue3,
        'output_prefix': args.output_prefix,
    }

    if args.config_file:
        file_config = load_config_from_file(args.config_file)
        config.update(file_config)

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    collator = BoltzCollator(config)
    collator.run_collation()


if __name__ == "__main__":
    main()
