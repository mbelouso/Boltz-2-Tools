'''
Program to filter Boltz-2 results based on various criteria such as confidence scores, affinity predictions, and binding site distances.

Usage:
python Filter_Boltz2.py --confidence-value 0.85 --affinity-value -1.5 --probability-binary-value 0.6 --binding-site-residue1 157 --binding-site-residue2 None --binding-site-residue3 None --distance-threshold 15.0 --config-file config.json --output-prefix boltz_results

Sample config.json:

{
    "confidence_value": 0.9,
    "affinity_value": -2.0,
    "probability_binary_value": 0.7,
    "binding_site_residue1": 120,
    "binding_site_residue2": 150,
    "distance_threshold": 12.0
}

The script will need to be copied to the Boltz-2 base directory and run from there with the appropriate anaconda environment activated.

'''

# Imports

import os
import pandas as pd
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from functools import partial
import time
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import modelcif.reader
import modelcif
import biotite
import biotite.structure.io as strucio
import biotite.structure as structure
import biotite.structure.io.pdb as pdb
from Bio.PDB import *
from Bio.PDB.Atom import Atom
import sys
import argparse
import yaml


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Functions________________________________________


# Function to parse Boltz2 results from a directory

def parse_arguments():
    """Parse command line arguments for configuration options."""
    parser = argparse.ArgumentParser(
        description="Process and filter Boltz-2 results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--confidence-value', 
        type=float, 
        default=0.85,
        help='Minimum confidence score threshold'
    )
    
    parser.add_argument(
        '--affinity-value', 
        type=float, 
        default=-1.5,
        help='Maximum affinity prediction value threshold'
    )
    
    parser.add_argument(
        '--probability-binary-value', 
        type=float, 
        default=0.6,
        help='Minimum affinity probability binary threshold'
    )
    
    parser.add_argument(
        '--binding-site-residue1', 
        type=int, 
        default=157,
        help='Primary binding site residue number'
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
        '--distance-threshold', 
        type=float, 
        default=15.0,
        help='Maximum distance threshold (Angstroms) for binding site filtering'
    )

    parser.add_argument(
        '--plot-residue1', 
        type=int, 
        default=None, 
        help='Residue index for distance plot #1'
    )

    parser.add_argument(
        '--plot-residue2', 
        type=int, 
        default=None, 
        help='Residue index for distance plot #2'
    )

    parser.add_argument(
        '--plot-residue3', 
        type=int, 
        default=None, 
        help='Residue index for distance plot #3'
    )
    
    parser.add_argument(
        '--plot-residue4', 
        type=int, 
        default=None, 
        help='Residue index for distance plot #4'
    )

    parser.add_argument(
        '--config-file', 
        type=str, 
        help='JSON configuration file (overrides command line arguments)'
    )
    
    parser.add_argument(
        '--output-prefix', 
        type=str, 
        default='boltz_results',
        help='Prefix for output files'
    )

    parser.add_argument(
    '--allow-empty-affinity',
    action='store_true',
    help='Allow filtering models even if affinity_pred_value and affinity_probability_binary are missing'
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


# Function to combine the CSV results from multiple files:
def combine_csv_results(file_list):
    combined_df = pd.concat([pd.read_csv(f) for f in file_list if f.endswith('_results.csv')])
    return combined_df

# Function to filter by distance and position. The inputs need to be up to three amino acid positions a distance per amino acid position
def filter_by_sites(distances, site1, site2, site3, distance_thresh):
    """
    Function to filter the dataset by defining 3 CA residue positions and distance threshold

    Parameters:
    distances (np.array): distances in 1D array
    site1, site2, site3 (int): refer to the row position and the hence the residue position
    distance_thresh (float): threshold for filtering by distance
    
    Returns: Boolean and distances to site
    """
    # read np array
    
    # Calculate the average distance to the specified sites
    if site2 is None:
        distance = np.array([distances[site1]])
    elif site3 is None:
        distance = np.array([distances[site1], distances[site2]])
    else:
        distance = np.array([distances[site1], distances[site2], distances[site3]])
    distance_average = np.mean(distance)  # Calculate the average distance
    # Check if all distances are below the threshold

    if distance_average < distance_thresh:
        return True
    else:
        return False

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
    
    #print(f"Chain A donors (N,O): {len(chain_a_donors)}")
    #print(f"Chain A acceptors (N,O,S,F): {len(chain_a_acceptors)}")
    #print(f"Chain B donors (N,O): {len(chain_b_donors)}")
    #print(f"Chain B acceptors (N,O,S,F): {len(chain_b_acceptors)}")
    
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
    """
    if len(hbonds) > 0:
        print("\nPotential hydrogen bond details:")
        for i, bond in enumerate(bond_details):
            print(f"  Bond {i+1}:")
            print(f"    Donor: {bond['donor_atom']} in {bond['donor_res']}{bond['donor_res_id']} (Chain {bond['donor_chain']})")
            print(f"    Acceptor: {bond['acceptor_atom']} in {bond['acceptor_res']}{bond['acceptor_res_id']} (Chain {bond['acceptor_chain']})")
            print(f"    Distance: {bond['distance']:.2f} Å")
            if bond.get('angle') is not None:
                print(f"    Angle: {bond['angle']:.2f}°")
            print()
        
        # Summary statistics
        distances = [bond['distance'] for bond in bond_details]
        print(f"Distance statistics:")
        print(f"  Mean: {np.mean(distances):.2f} Å")
        print(f"  Min: {np.min(distances):.2f} Å")
        print(f"  Max: {np.max(distances):.2f} Å")
        
        # Count by residue types
        donor_residues = [bond['donor_res'] for bond in bond_details]
        acceptor_residues = [bond['acceptor_res'] for bond in bond_details]
        
        unique_donors, donor_counts = np.unique(donor_residues, return_counts=True)
        unique_acceptors, acceptor_counts = np.unique(acceptor_residues, return_counts=True)
        
        print(f"\nDonor residue types: {dict(zip(unique_donors, donor_counts))}")
        print(f"Acceptor residue types: {dict(zip(unique_acceptors, acceptor_counts))}")
    """
    return len(hbonds), bond_details

def process_hydrogen_bonds(file_path):
    """Process a single PDB file for hydrogen bond analysis"""
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

        # Return the result as a Series with the file path as name
        return pd.Series(num_hbonds, name=file_path)
    except FileNotFoundError as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def align_structure(pdb_file, reference_ca_coords, directory_to_parse, directory_to_save):
    """
    Function to align a single PDB structure against reference CA coordinates
    """
    try:
        parser = PDBParser(QUIET=True)
        super_imposer = Superimposer()
        
        # Load the structure
        structure = parser.get_structure(pdb_file, os.path.join(directory_to_parse, pdb_file))
        
        # Get CA atoms from current structure
        ca_atoms_current = [atom for atom in structure.get_atoms() if atom.get_id() == 'CA']
        
        if len(reference_ca_coords) != len(ca_atoms_current):
            return f"Error: The number of CA atoms in '{pdb_file}' does not match the reference structure. Reference: {len(reference_ca_coords)}, Target: {len(ca_atoms_current)}"
        
        if len(ca_atoms_current) < 3:
            return f"Error: At least 3 CA atoms are required for alignment in '{pdb_file}'."
        
        # Create dummy atoms for reference coordinates (needed for Superimposer)
        # We'll use the coordinates from the reference structure
        
        reference_atoms = []
        for i, coord in enumerate(reference_ca_coords):
            atom = Atom('CA', coord, 0, 1, ' ', 'CA', i, 'C')
            reference_atoms.append(atom)
        
        # Perform alignment
        super_imposer.set_atoms(reference_atoms, ca_atoms_current)
        super_imposer.apply(structure.get_atoms())
        
        # Get RMSD for reporting
        rmsd = super_imposer.rms
        
        # Save the aligned structure
        io = PDBIO()
        io.set_structure(structure)
        output_file = os.path.join(directory_to_save, f"aligned_{pdb_file}")
        io.save(output_file)
        
        return f"Aligned structure saved to {output_file} (RMSD: {rmsd:.2f} Å)"
        
    except Exception as e:
        return f"Error processing {pdb_file}: {str(e)}"


# End of Functions________________________________________


class BoltzResultsProcessor:
    """Main class for processing Boltz-2 results with filtering and analysis."""
    
    def __init__(self, config: dict):
        self.config = config
        self.combined_df = None
        self.hydrogen_bond_df = None
        self.distance_df = None
        self.pdb_file_list = []
        
    def setup_directories(self) -> None:
        """Create necessary output directories."""
        directories = [
            'combined_models', 'distance_filtered', 'filtered_models', 
            'filtered_aligned', 'temp_archive'
        ]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
    def get_boltz_folders(self) -> List[str]:
        """Get list of Boltz result folders."""
        return [f for f in os.listdir('./') if f.startswith('boltz_results_')]
    
    def _normalize_path(self, path: str) -> str:
        """Normalize file paths for consistent merging."""
        # Convert to Path object for consistent handling
        path_obj = Path(path)
        
        # If it's an absolute path, make it relative to current directory
        if path_obj.is_absolute():
            try:
                path_obj = path_obj.relative_to(Path.cwd())
            except ValueError:
                # If can't make relative, keep as is
                pass
        
        # Ensure it starts with "./" for consistency
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
            predictions_folder = Path(folder) / 'predictions' / folder_id
            
            # Check if the folder exists; if not, try the full folder name
            if not predictions_folder.exists():
                logger.warning(f"Predictions folder does not exist: {predictions_folder}")
                predictions_folder = Path(folder) / 'predictions' / folder_id  # Use the full folder name
                if not predictions_folder.exists():
                    logger.error(f"Predictions folder still does not exist: {predictions_folder}")
                    continue
        
            logger.info(f"Processing folder: {predictions_folder}")
            results_df = parse_boltz2_results(str(predictions_folder))
            csv_filename = f"{folder}_results.csv"
            results_df.to_csv(csv_filename, index=False)
            csv_files.append(csv_filename)
    
        # Combine all CSV files
        self.combined_df = combine_csv_results(csv_files)
        
        # Normalize paths in combined_df - check if column exists first
        if 'model_path' in self.combined_df.columns:
            self.combined_df['model_path'] = self.combined_df['model_path'].apply(self._normalize_path)
            
            # Debug: Show sample paths
            logger.info(f"Sample combined paths: {list(self.combined_df['model_path'][:3])}")
        else:
            logger.error("No 'model_path' column found in combined DataFrame")
            logger.info(f"Available columns: {list(self.combined_df.columns)}")
        
        self.combined_df.to_csv('boltz_results_combined.csv', index=False)
        logger.info("Combined results saved to 'boltz_results_combined.csv'")
        
        # Clean up temporary files
        self._cleanup_temp_files(csv_files)
        
        return self.combined_df
    
    def _cleanup_temp_files(self, file_list: List[str]) -> None:
        """Remove temporary CSV files."""
        for file in file_list:
            if Path(file).exists():
                os.remove(file)
    
    def copy_pdb_files(self) -> None:
        """Copy PDB files to combined_models directory."""
        logger.info("Copying PDB files to combined_models directory...")
        combined_output_dir = Path('combined_models')
        
        for file_path in self.combined_df['model_path'].unique():
            if Path(file_path).exists():
                shutil.copy2(file_path, combined_output_dir)
                logger.debug(f"Copied {file_path} to {combined_output_dir}")
            else:
                logger.warning(f"File does not exist: {file_path}")
    
    def collect_pdb_files(self) -> List[str]:
        """Collect all PDB file paths from both possible prediction subfolders."""
        self.pdb_file_list = []
        folder_list = self.get_boltz_folders()
        
        for folder in folder_list:
            # Extract folder ID and full folder name
            folder_id = folder.split('_', 2)[2]
            predictions_folder1 = Path(folder) / 'predictions' / folder_id
            predictions_folder2 = Path(folder) / 'predictions' / folder  # Use full folder name
            
            # Check both possible prediction folders
            for predictions_folder in [predictions_folder1, predictions_folder2]:
                if predictions_folder.exists():
                    pdb_files = list(predictions_folder.glob('*.pdb'))
                    self.pdb_file_list.extend([str(f) for f in pdb_files])
    
        logger.info(f"Found {len(self.pdb_file_list)} PDB files to process")
        return self.pdb_file_list
    
    def calculate_hydrogen_bonds(self) -> pd.DataFrame:
        """Calculate hydrogen bonds using parallel processing."""
        logger.info("Starting hydrogen bond analysis...")
        
        if not self.pdb_file_list:
            self.collect_pdb_files()
        
        if not self.pdb_file_list:
            logger.error("No PDB files found for hydrogen bond analysis.")
            return pd.DataFrame()
        
        start_time = time.time()
        num_processes = max(1, min(mp.cpu_count(), len(self.pdb_file_list)))  # Ensure at least 1 process
        
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_hydrogen_bonds, self.pdb_file_list)
        
        # Filter successful results
        hydrogen_bond_data = [result for result in results if result is not None]
        
        end_time = time.time()
        logger.info(f"Hydrogen bond analysis completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Successfully processed {len(hydrogen_bond_data)} files")
        
        if not hydrogen_bond_data:
            logger.warning("No hydrogen bond data was successfully processed")
            return pd.DataFrame()
        
        # Create DataFrame
        self.hydrogen_bond_df = pd.concat(hydrogen_bond_data, axis=1).transpose()
        self.hydrogen_bond_df.columns = ['num_hbonds']
        self.hydrogen_bond_df.index.name = 'model_path'
        
        # Normalize paths - this is crucial for proper merging
        self.hydrogen_bond_df.index = self.hydrogen_bond_df.index.map(
            lambda x: self._normalize_path(x)
        )
        
        # Debug: Show sample paths
        logger.info(f"Sample hydrogen bond paths: {list(self.hydrogen_bond_df.index[:3])}")
        
        self.hydrogen_bond_df.to_csv('hydrogen_bond_data.csv')
        logger.info("Hydrogen bond data saved to 'hydrogen_bond_data.csv'")
        
        return self.hydrogen_bond_df
    
    def calculate_distances(self) -> pd.DataFrame:
        """Calculate distances with parallel processing."""
        logger.info("Calculating distances...")
        
        if not self.pdb_file_list:
            self.collect_pdb_files()
        
        # Process distance calculations
        distance_data = self._process_distance_calculations()
        
        if not distance_data:
            logger.warning("No distance data was successfully processed")
            return pd.DataFrame()
        
        # Create distance DataFrame
        self.distance_df = pd.concat(distance_data, axis=1).transpose()
        self.distance_df.to_csv('distance_matrices.csv')
        
        # Normalize paths and reset index
        self.distance_df.index.name = 'model_path'
        self.distance_df.index = self.distance_df.index.map(
            lambda x: self._normalize_path(x)
        )
        
        # Debug: Show sample paths
        logger.info(f"Sample distance paths: {list(self.distance_df.index[:3])}")
        
        self.distance_df = self.distance_df.reset_index()
        
        return self.distance_df
    
    def _process_distance_calculations(self) -> List[pd.Series]:
        """Process distance calculations for all PDB files."""
        distance_data = []
        distance_filtered_dir = Path('distance_filtered')
        
        for file_path in self.pdb_file_list:
            if not Path(file_path).is_file():
                continue
            
            try:
                distances = self._calculate_file_distances(file_path)
                distance_data.append(pd.Series(distances, name=file_path))
                
                # Filter and copy if meets criteria
                if self._passes_distance_filter(distances):
                    shutil.copy2(file_path, distance_filtered_dir)
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        return distance_data
    
    def _calculate_file_distances(self, file_path: str) -> List[float]:
        """Calculate distances for a single file."""
        pdb_file_obj = pdb.PDBFile.read(file_path)
        pdb_data = pdb_file_obj.get_structure()
        
        CA_positions = extract_ca_positions(pdb_data, chain_id='A')
        Lig_positions = extract_atoms_by_chain(pdb_data, chain_id='B')
        Lig_COM = calculate_center_of_mass(Lig_positions)
        
        return calculate_distances_to_com(CA_positions, Lig_COM)
    
    def _passes_distance_filter(self, distances: List[float]) -> bool:
        """Check if distances pass the filtering criteria."""
        return filter_by_sites(
            distances,
            self.config['binding_site_residue1'],
            self.config.get('binding_site_residue2'),
            self.config.get('binding_site_residue3'),
            self.config['distance_threshold']
        )
    
    def create_reduced_distance_df(self) -> pd.DataFrame:
        """Create reduced distance DataFrame with specific residue."""
        if self.distance_df is None or self.distance_df.empty:
            logger.error("Distance DataFrame is empty or None")
            return pd.DataFrame()
        
        binding_site_residue_column = self.config['binding_site_residue1']
        
        # Check if the column exists
        if binding_site_residue_column in self.distance_df.columns:
            logger.info(f"Found binding site residue column: {binding_site_residue_column}")
            
            # Create reduced DataFrame
            reduced_distance_df = self.distance_df.groupby('model_path', as_index=False)[
                binding_site_residue_column
            ].mean()
            reduced_distance_df.rename(
                columns={binding_site_residue_column: 'distance_to_residue'}, 
                inplace=True
            )
            
            # Export reduced distance data
            reduced_distance_df.to_csv('reduced_distance_data.csv', index=False)
            logger.info(f"Reduced distance data saved with {len(reduced_distance_df)} rows")
            
            return reduced_distance_df
        else:
            logger.error(f"Column '{binding_site_residue_column}' not found in distance_df")
            logger.info(f"Available columns: {list(self.distance_df.columns)}")
            return pd.DataFrame()
    
    def merge_all_data(self) -> pd.DataFrame:
        """Merge hydrogen bond and distance data with combined DataFrame."""
        logger.info("Merging all data...")
        
        # Check if DataFrames exist before merging
        if self.combined_df is None or self.combined_df.empty:
            logger.error("Combined DataFrame is None or empty. Run parse_boltz_results() first.")
            return pd.DataFrame()
        
        original_shape = self.combined_df.shape
        logger.info(f"Starting with combined DataFrame shape: {original_shape}")
        
        # Merge hydrogen bonds if available
        if self.hydrogen_bond_df is not None and not self.hydrogen_bond_df.empty:
            logger.info("Merging hydrogen bond data...")
            
            # Debug merge
            logger.info(f"Combined DF unique paths: {self.combined_df['model_path'].nunique()}")
            logger.info(f"H-bond DF unique paths: {self.hydrogen_bond_df.index.nunique()}")
            
            # Check for overlap
            overlap = set(self.combined_df['model_path']).intersection(set(self.hydrogen_bond_df.index))
            logger.info(f"Path overlap between DataFrames: {len(overlap)} paths")
            
            if len(overlap) == 0:
                logger.error("No overlapping paths found between combined and hydrogen bond DataFrames!")
                logger.info(f"Sample combined paths: {list(self.combined_df['model_path'][:5])}")
                logger.info(f"Sample h-bond paths: {list(self.hydrogen_bond_df.index[:5])}")
            
            self.combined_df = pd.merge(
                self.combined_df, 
                self.hydrogen_bond_df, 
                left_on='model_path', 
                right_index=True, 
                how='left'
            )
            logger.info(f"After h-bond merge. DataFrame shape: {self.combined_df.shape}")
            
            # Check for NaN values
            nan_count = self.combined_df['num_hbonds'].isna().sum()
            logger.info(f"NaN values in num_hbonds: {nan_count}")
            
        else:
            logger.warning("No hydrogen bond data to merge")
        
        # Merge distances if available
        if self.distance_df is not None and not self.distance_df.empty:
            logger.info("Merging distance data...")
            reduced_distance_df = self.create_reduced_distance_df()
            
            if not reduced_distance_df.empty:
                # Debug merge
                logger.info(f"Combined DF unique paths: {self.combined_df['model_path'].nunique()}")
                logger.info(f"Distance DF unique paths: {reduced_distance_df['model_path'].nunique()}")
                
                # Check for overlap
                overlap = set(self.combined_df['model_path']).intersection(set(reduced_distance_df['model_path']))
                logger.info(f"Path overlap for distance merge: {len(overlap)} paths")
                
                if len(overlap) == 0:
                    logger.error("No overlapping paths found for distance merge!")
                    logger.info(f"Sample combined paths: {list(self.combined_df['model_path'][:5])}")
                    logger.info(f"Sample distance paths: {list(reduced_distance_df['model_path'][:5])}")
                
                self.combined_df = pd.merge(
                    self.combined_df, 
                    reduced_distance_df, 
                    on='model_path', 
                    how='left'
                )
                logger.info(f"After distance merge. DataFrame shape: {self.combined_df.shape}")
                
                # Check for NaN values
                nan_count = self.combined_df['distance_to_residue'].isna().sum()
                logger.info(f"NaN values in distance_to_residue: {nan_count}")
                
            else:
                logger.warning("Failed to create reduced distance DataFrame")
        else:
            logger.warning("No distance data to merge")
        
        # Export the final combined DataFrame
        output_file = 'boltz_results_combined_with_hbonds_distances.csv'
        try:
            self.combined_df.to_csv(output_file, index=False)
            logger.info(f"Combined DataFrame exported to '{output_file}'")
            logger.info(f"Final DataFrame shape: {self.combined_df.shape}")
            logger.info(f"Columns: {list(self.combined_df.columns)}")
            
            # Show merge statistics
            if 'num_hbonds' in self.combined_df.columns:
                valid_hbonds = self.combined_df['num_hbonds'].notna().sum()
                logger.info(f"Valid hydrogen bond values: {valid_hbonds}/{len(self.combined_df)}")
            
            if 'distance_to_residue' in self.combined_df.columns:
                valid_distances = self.combined_df['distance_to_residue'].notna().sum()
                logger.info(f"Valid distance values: {valid_distances}/{len(self.combined_df)}")
            
            # Show a sample of the data with non-NaN values if available
            non_nan_data = self.combined_df.dropna(subset=['num_hbonds', 'distance_to_residue'])
            if not non_nan_data.empty:
                logger.info(f"Sample merged data (non-NaN):\n{non_nan_data[['model_path', 'num_hbonds', 'distance_to_residue']].head()}")
            else:
                logger.warning("No rows with complete data found!")
                
        except Exception as e:
            logger.error(f"Failed to export combined DataFrame: {e}")
        
        return self.combined_df

    def filter_results(self) -> pd.DataFrame:
        """Filter results based on all criteria."""
        logger.info("Filtering results based on criteria...")
    
        required_filter_columns = ['num_hbonds', 'distance_to_residue']
        missing_columns = [col for col in required_filter_columns if col not in self.combined_df.columns]
    
        if missing_columns:
            logger.error(f"Missing required columns for filtering: {missing_columns}")
            return pd.DataFrame()
        
        valid_data_mask = self.combined_df[required_filter_columns].notna().all(axis=1)
        valid_data_count = valid_data_mask.sum()
        
        logger.info(f"Rows with complete data for filtering: {valid_data_count}/{len(self.combined_df)}")
        
        if valid_data_count == 0:
            logger.error("No rows with complete data available for filtering!")
            return pd.DataFrame()
        
        config = self.config
        # If allow_empty_affinity is set, skip affinity filters if columns are missing or NaN
        if config.get('allow_empty_affinity', False):
            filter_mask = (
                (self.combined_df['confidence_score'] > config['confidence_value']) &
                (self.combined_df['distance_to_residue'] < config['distance_threshold']) &
                (self.combined_df['num_hbonds'] > 0) &
                valid_data_mask
            )
        else:
            filter_mask = (
                (self.combined_df['confidence_score'] > config['confidence_value']) &
                (self.combined_df['affinity_pred_value'] < config['affinity_value']) &
                (self.combined_df['distance_to_residue'] < config['distance_threshold']) &
                (self.combined_df['num_hbonds'] > 0) &
                (self.combined_df['affinity_probability_binary'] > config['probability_binary_value']) &
                valid_data_mask
            )
        
        filtered_df = self.combined_df[filter_mask]
        filtered_df.to_csv('Final_Filtered_models.csv', index=False)
        logger.info(f"Filtered {len(filtered_df)} models from {valid_data_count} models with complete data")
        logger.info(f"Filter criteria applied:")
        logger.info(f"  - confidence_score > {config['confidence_value']}")
        logger.info(f"  - affinity_pred_value < {config['affinity_value']}")
        logger.info(f"  - distance_to_residue < {config['distance_threshold']}")
        logger.info(f"  - num_hbonds > 0")
        logger.info(f"  - affinity_probability_binary > {config['probability_binary_value']}")
        
        # Copy filtered models
        if not filtered_df.empty:
            self._copy_filtered_models(filtered_df)
        
        return filtered_df
    
    def _copy_filtered_models(self, filtered_df: pd.DataFrame) -> None:
        """Copy filtered model files to output directory."""
        filtered_dir = Path('filtered_models')
        
        for _, row in filtered_df.iterrows():
            model_path = Path(row['model_path'])
            if model_path.exists():
                shutil.copy2(model_path, filtered_dir)
                logger.debug(f"Copied {model_path} to {filtered_dir}")
            else:
                logger.warning(f"Model file does not exist: {model_path}")

    def align_structures(self, directory_to_parse: str = './filtered_models', 
                        directory_to_save: str = './filtered_aligned') -> dict:
        """Align filtered structures using parallel processing."""
        logger.info("Starting parallel PDB alignment...")
        
        directory_to_parse = Path(directory_to_parse)
        directory_to_save = Path(directory_to_save)
        directory_to_save.mkdir(exist_ok=True)
        
        # Validation
        if not directory_to_parse.exists():
            raise FileNotFoundError(f"Directory '{directory_to_parse}' does not exist")
        
        pdb_files = list(directory_to_parse.glob('*.pdb'))
        if len(pdb_files) < 2:
            raise ValueError("At least two PDB files are required for alignment")
        
        logger.info(f"Found {len(pdb_files)} PDB files to process")
        
        # Setup reference structure
        reference_structure, reference_ca_coords = self._setup_reference_structure(
            directory_to_parse, pdb_files[0], directory_to_save
        )
        
        # Parallel alignment
        start_time = time.time()
        num_processes = min(mp.cpu_count(), len(pdb_files) - 1)
        
        align_func = partial(
            align_structure,
            reference_ca_coords=reference_ca_coords,
            directory_to_parse=str(directory_to_parse),
            directory_to_save=str(directory_to_save)
        )
        
        files_to_process = [f.name for f in pdb_files[1:]]
        
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(align_func, files_to_process)
        
        # Process results
        stats = self._process_alignment_results(results, start_time, len(pdb_files), num_processes)
        
        return stats
    
    def _setup_reference_structure(self, directory_to_parse: Path, reference_file: Path, 
                                 directory_to_save: Path) -> Tuple:
        """Setup reference structure for alignment."""
        from Bio.PDB import PDBParser, PDBIO
        
        parser = PDBParser(QUIET=True)
        reference_structure = parser.get_structure("reference", reference_file)
        reference_ca_atoms = [atom for atom in reference_structure.get_atoms() if atom.get_id() == 'CA']
        reference_ca_coords = [atom.get_coord() for atom in reference_ca_atoms]
        
        logger.info(f"Reference structure: {reference_file.name} with {len(reference_ca_coords)} CA atoms")
        
        # Save reference structure
        io = PDBIO()
        io.set_structure(reference_structure)
        output_file_ref = directory_to_save / f"aligned_{reference_file.name}"
        io.save(str(output_file_ref))
        
        return reference_structure, reference_ca_coords
    
    def _process_alignment_results(self, results: List[str], start_time: float, 
                                 total_files: int, num_processes: int) -> dict:
        """Process and log alignment results."""
        successful_alignments = 0
        failed_alignments = 0
        rmsd_values = []
        
        for result in results:
            if "Error" in result:
                failed_alignments += 1
                logger.error(result)
            else:
                successful_alignments += 1
                logger.debug(result)
                
                # Extract RMSD values
                if "RMSD:" in result:
                    try:
                        rmsd_str = result.split("RMSD: ")[1]
                        rmsd_values.append(float(rmsd_str))
                    except:
                        continue
        
        end_time = time.time()
        
        # Log statistics
        logger.info(f"Alignment process completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Processed {total_files} files using {num_processes} parallel processes")
        logger.info(f"Successful alignments: {successful_alignments}")
        logger.info(f"Failed alignments: {failed_alignments}")
        
        if rmsd_values:
            stats = {
                'mean_rmsd': np.mean(rmsd_values),
                'min_rmsd': np.min(rmsd_values),
                'max_rmsd': np.max(rmsd_values),
                'std_rmsd': np.std(rmsd_values),
                'successful_alignments': successful_alignments,
                'failed_alignments': failed_alignments,
                'processing_time': end_time - start_time
            }
            
            logger.info("Alignment Statistics:")
            logger.info(f"  Mean RMSD: {stats['mean_rmsd']:.2f} Å")
            logger.info(f"  Min RMSD: {stats['min_rmsd']:.2f} Å")
            logger.info(f"  Max RMSD: {stats['max_rmsd']:.2f} Å")
            logger.info(f"  Std RMSD: {stats['std_rmsd']:.2f} Å")
            
            return stats
        
        return {
            'successful_alignments': successful_alignments,
            'failed_alignments': failed_alignments,
            'processing_time': end_time - start_time
        }
    
    def create_plots(self) -> None:
        """Create all analysis plots."""
        logger.info("Creating analysis plots...")
        
        # Distribution plots
        self._create_distribution_plots()
        
        # Distance plots
        self._create_distance_plots()
        
        # Correlation plots
        self._create_correlation_plots()
    
    def _create_distribution_plots(self) -> None:
        """Create distribution plots for main metrics."""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.histplot(self.combined_df['confidence_score'], bins=30, kde=True)
        plt.title('Distribution of Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        sns.histplot(self.combined_df['affinity_pred_value'], bins=30, kde=True)
        plt.title('Distribution of Affinity Prediction Values')
        plt.xlabel('Affinity Prediction Value')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 3)
        sns.histplot(self.combined_df['affinity_probability_binary'], bins=30, kde=True)
        plt.title('Distribution of Affinity Probability Binary')
        plt.xlabel('Affinity Probability Binary')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('boltz_results_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_distance_plots(self) -> None:
        """Create distance-related plots."""
        if Path('distance_matrices.csv').exists():
            distance_data = pd.read_csv('distance_matrices.csv')

            # Build a map from integer residue IDs -> actual column names in CSV
            col_map = {}
            for col in distance_data.columns:
                # Skip non-distance columns like 'model_path'
                if col == 'model_path':
                    continue
                try:
                    col_map[int(col)] = col  # headers saved as strings
                except (ValueError, TypeError):
                    # If headers are already ints, or not numeric, handle gracefully
                    if isinstance(col, int):
                        col_map[col] = col

            residues = [
                self.config.get('plot-residue1'),
                self.config.get('plot-residue2'),
                self.config.get('plot-residue3'),
                self.config.get('plot-residue4')
            ]
            residues = [r for r in residues if r is not None]

            if not residues:
                logger.warning("No valid residues specified for distance plots. Skipping.")
                return

            plotted = 0
            plt.figure(figsize=(12, 10))
            for i, residue in enumerate(residues, 1):
                if residue in col_map:
                    plt.subplot(2, 2, i)
                    col_name = col_map[residue]
                    sns.histplot(distance_data[col_name].dropna(), bins=50, kde=True)
                    plt.title(f'Distribution of Distances to Residue {residue}')
                    plt.xlabel('Distance to Residue (Å)')
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                    plotted += 1
                else:
                    logger.warning(f"Residue {residue} not found in distance matrix columns. Available numeric columns: "
                                   f"{sorted(list(col_map.keys()))[:10]}{' ...' if len(col_map)>10 else ''}")

            if plotted == 0:
                logger.warning("No matching residue columns found. Skipping distance plots.")
                plt.close()
                return

            plt.tight_layout()
            plt.savefig('distance_distribution_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logger.warning("distance_matrices.csv not found. Skipping distance plots.")
    
    def _create_correlation_plots(self) -> None:
        """Create correlation plots."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.combined_df, 
            x='affinity_pred_value', 
            y='num_hbonds', 
            hue='confidence_score', 
            palette='viridis', 
            alpha=0.7
        )
        plt.title('Number of Hydrogen Bonds vs. Affinity Prediction Value')
        plt.xlabel('Affinity Prediction Value')
        plt.ylabel('Number of Hydrogen Bonds')
        plt.legend(title='Confidence Score')
        plt.savefig('hbonds_vs_affinity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_archive(self) -> None:
        """Create ZIP archive of results."""
        logger.info("Creating results archive...")
        
        config = self.config
        archive_name = f'boltz_results_archive_confidence{config["confidence_value"]}_affinity{config["affinity_value"]}'
        temp_dir = Path('temp_archive')
        temp_dir.mkdir(exist_ok=True)
        
        # Copy files and directories
        files_to_copy = [
            'boltz_results_combined_with_hbonds_distances.csv',
            'Final_Filtered_models.csv'
        ]
        
        for file in files_to_copy:
            if Path(file).exists():
                shutil.copy2(file, temp_dir)
        
        # Copy directories
        dirs_to_copy = ['combined_models', 'filtered_aligned']
        for directory in dirs_to_copy:
            src_dir = Path(directory)
            if src_dir.exists():
                shutil.copytree(src_dir, temp_dir / directory, dirs_exist_ok=True)
        
        # Copy PNG files
        for png_file in Path('.').glob('*.png'):
            shutil.copy2(png_file, temp_dir)
        
        # Create archive
        shutil.make_archive(archive_name, 'zip', temp_dir)
        shutil.rmtree(temp_dir)
        
        logger.info(f"Results archived to {archive_name}.zip")
    
    def run_full_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        logger.info("Starting full Boltz-2 analysis pipeline...")
        
        try:
            # Setup
            self.setup_directories()
            
            # Parse results
            self.parse_boltz_results()
            if self.combined_df is None or self.combined_df.empty:
                raise ValueError("Failed to parse Boltz results - no data found")
            
            logger.info(f"Initial combined DataFrame shape: {self.combined_df.shape}")
            
            # Copy PDB files
            self.copy_pdb_files()
            
            # Calculate metrics
            self.calculate_hydrogen_bonds()
            self.calculate_distances()
            
            # Export data summary for debugging
            self.export_data_summary()
            
            # Merge and filter
            merged_df = self.merge_all_data()
            if merged_df.empty:
                raise ValueError("Failed to merge data - resulting DataFrame is empty")
            
            # Filter results
            filtered_df = self.filter_results()
            
            # Align structures if we have filtered results
            if len(filtered_df) > 1:
                self.align_structures()
            else:
                logger.warning("Not enough filtered models for alignment")
            
            # Create visualizations
            self.create_plots()
            
            # Archive results
            self.create_archive()
            
            logger.info("Analysis pipeline completed successfully!")
            logger.info(f"Final results: {len(filtered_df)} filtered models from {len(self.combined_df)} total")
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            # Export what we have for debugging
            if self.combined_df is not None:
                self.combined_df.to_csv('debug_combined_df.csv', index=False)
                logger.info("Debug DataFrame saved to 'debug_combined_df.csv'")
            raise

    def export_data_summary(self) -> None:
        """Export a summary of all data for debugging."""
        logger.info("Creating data summary...")
        
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'config': self.config,
            'combined_df_shape': self.combined_df.shape if self.combined_df is not None else None,
            'combined_df_columns': list(self.combined_df.columns) if self.combined_df is not None else None,
            'hydrogen_bond_df_shape': self.hydrogen_bond_df.shape if self.hydrogen_bond_df is not None else None,
            'distance_df_shape': self.distance_df.shape if self.distance_df is not None else None,
            'pdb_files_count': len(self.pdb_file_list)
        };
        
        # Save summary as JSON
        with open('data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Data summary saved to 'data_summary.json'")

    def validate_data_integrity(self) -> bool:
        """Validate that all data is properly loaded and merged."""
        logger.info("Validating data integrity...")
        
        issues = []
        
        # Check combined_df
        if self.combined_df is None or self.combined_df.empty:
            issues.append("Combined DataFrame is empty or None")
        else:
            logger.info(f"Combined DataFrame: {self.combined_df.shape} rows x {self.combined_df.shape[1]} columns")
        
        # Check hydrogen_bond_df
        if self.hydrogen_bond_df is None or self.hydrogen_bond_df.empty:
            issues.append("Hydrogen bond DataFrame is empty or None")
        else:
            logger.info(f"Hydrogen bond DataFrame: {self.hydrogen_bond_df.shape}")
        
        # Check distance_df
        if self.distance_df is None or self.distance_df.empty:
            issues.append("Distance DataFrame is empty or None")
        else:
            logger.info(f"Distance DataFrame: {self.distance_df.shape}")
        
        # Check for required columns
        if self.combined_df is not None:
            required_columns = ['model_path', 'confidence_score', 'affinity_pred_value', 'affinity_probability_binary']
            missing_columns = [col for col in required_columns if col not in self.combined_df.columns]
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")
        
        if issues:
            logger.error("Data integrity issues found:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("Data integrity validation passed")
        return True


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Start with default config
    config = {
        'allow_empty_affinity': args.allow_empty_affinity,
        'confidence_value': args.confidence_value,
        'affinity_value': args.affinity_value,
        'probability_binary_value': args.probability_binary_value,
        'binding_site_residue1': args.binding_site_residue1,
        'binding_site_residue2': args.binding_site_residue2,
        'binding_site_residue3': args.binding_site_residue3,
        'distance_threshold': args.distance_threshold,
        'output_prefix': getattr(args, 'output_prefix', 'boltz_results'),
        'plot_residue1': args.plot_residue1,
        'plot_residue2': args.plot_residue2,
        'plot_residue3': args.plot_residue3,
        'plot_residue4': args.plot_residue4,
    }
    
    # Override with config file if provided
    if args.config_file:
        file_config = load_config_from_file(args.config_file)
        config.update(file_config)
    
    # Log the configuration being used
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    processor = BoltzResultsProcessor(config)
    
    # Run with validation
    processor.run_full_analysis()
    
    # Validate final result
    if processor.validate_data_integrity():
        logger.info("All data successfully processed and exported")
    else:
        logger.error("Data integrity issues detected")

if __name__ == "__main__":
    main()
else:
    logger.warning("This script is intended to be run as a standalone program. Importing may not work as expected.")
    # If imported, we can still access the BoltzResultsProcessor class
    # and its methods, but the main execution will not run.
    # This allows for unit testing or importing in other scripts without executing the main function.
    # Example usage:
    # processor = BoltzResultsProcessor(config)
    # processor.run_full_analysis()