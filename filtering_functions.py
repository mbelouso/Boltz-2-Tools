# functions for analysis of Boltz2 results

import os
import pandas as pd
import json
import shutil
import numpy as np
import modelcif.reader
import modelcif
import biotite
import biotite.structure.io as strucio
import biotite.structure as structure
import biotite.structure.io.pdb as pdb
import numpy as np
from Bio.PDB import *
import sys

# Function to parse Boltz2 results from a directory
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
            #print(f"Loaded affinity data for key: '{model_key}' from file: {aff_file}")  # Debugging statement
        except Exception as e:
            print(f"Error loading affinity file {aff_file}: {e}")

    # Process confidence files and merge with affinity data
    for conf_file in confidence_files:
        model_index = conf_file.split('_')[-1].split('.')[0]  # Extract model index from filename
        base_name = conf_file.replace('confidence_', '').replace(f'_model_{model_index}.json', '')  # Extract base name
        model_file = f"{base_name}_model_{model_index}.pdb"  # Assuming the model files are in PDB format
        model_path = os.path.join(directory, model_file)
        model_key = f"{base_name}"  # Key matches affinity file base name

        # Load confidence data
        try:
            confidence_data = json.load(open(os.path.join(directory, conf_file), 'r'))
        except Exception as e:
            print(f"Error loading confidence file {conf_file}: {e}")
            continue
        
        #print(f"Processing confidence file: {conf_file}")
        #print(f"Generated model key: '{model_key}'")
        #print(f"Constructed model path: {model_path}")  # Debugging statement

        if os.path.exists(model_path):
                        
            # Create the result dictionary with confidence data
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
                #print(f"✓ Successfully added affinity data for: {model_key}")  # Debugging statement
            else:
                # Add None values for affinity data if not available
                result_entry.update({
                    'affinity_pred_value': None,
                    'affinity_probability_binary': None,
                    'affinity_pred_value1': None,
                    'affinity_probability_binary1': None,
                    'affinity_pred_value2': None,
                    'affinity_probability_binary2': None
                })
                print(f"✗ No affinity data found for: '{model_key}'")  # Debugging statement
                print(f"  Available keys: {list(affinity_dict.keys())}")
            
            results.append(result_entry)
    
        else:
            print(f"Model file does not exist: {model_path}")  # Debugging statement

    #print(f"\nProcessed {len(results)} models total")
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
        from Bio.PDB.Atom import Atom
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


