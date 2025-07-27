# functions for analysis of Boltz2 results

import os
import pandas as pd
import json
import shutil
import numpy as np
import modelcif.reader
import modelcif
import biotite
import biotite.structure as struc
import biotite.structure.io as strucio

def parse_boltz2_results(directory):
    results = []
    print(f"Scanning directory: {directory}")  # Debugging statement
    confidence_files = [f for f in os.listdir(directory) if f.startswith('confidence_') and f.endswith('.json')]
    print(f"Found confidence files: {confidence_files}")  # Debugging statement

    for conf_file in confidence_files:
        model_index = conf_file.split('_')[-1].split('.')[0]  # Extract model index from filename
        base_name = '_'.join(conf_file.split('_')[1:-2])
        confidence_data = json.load(open(os.path.join(directory, conf_file), 'r'))  # Load the JSON data

        # Find the corresponding model file dynamically based on the confidence file name
        base_name = conf_file.replace('confidence_', '').replace(f'_model_{model_index}.json', '')
        model_file = f"{base_name}_model_{model_index}.cif"
        model_path = os.path.join(directory, model_file)
        print(f"Constructed model path: {model_path}")  # Debugging statement

        if os.path.exists(model_path):
            print(f"Model file exists: {model_path}")  # Debugging statement
            results.append({
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
            })
        else:
            print(f"Model file does not exist: {model_path}")  # Debugging statement

    return pd.DataFrame(results)

# Function to combine the CSV results from multiple files:
def combine_csv_results(file_list):
    combined_df = pd.concat([pd.read_csv(f) for f in file_list if f.endswith('_results.csv')])
    return combined_df



def extract_coordinates(file_path, target_atom_id=None, target_asym_unit_id=None):
    # Open the CIF file and pass the file handle to the reader
    with open(file_path, 'r') as fh:
        systems = modelcif.reader.read(fh)  # Returns a list of System objects
    
    # print(f"Number of systems found: {len(systems)}") # Debugging line
    coordinates = None  # Initialize coordinates before the loop
    for system in systems:
        # print(f"System ID: {system.id}")
        
        # Extract models
        models = system._all_models()  # Call the method to retrieve models
        for model_tuple in models:  # Iterate over the generator
            model_list, model_object = model_tuple
            
            # Extract atoms using the `get_atoms` method
            atoms = model_object.get_atoms()
            for atom in atoms:
                # Retrieve atom_id and asym_unit.id
                atom_id = atom.atom_id
                asym_unit = atom.asym_unit
                asym_unit_id = getattr(asym_unit, 'id', None) if asym_unit else None
                
                # Apply filters
                if (target_atom_id is None or atom_id == target_atom_id) and \
                   (target_asym_unit_id is None or asym_unit_id == target_asym_unit_id):
                    # Extract the ouput coordinates
                    if coordinates is None:
                        coordinates = np.array([atom.x, atom.y, atom.z])
                    else:
                        coordinates = np.vstack((coordinates, np.array([atom.x, atom.y, atom.z])))
    return coordinates

def parse_boltz2_results(directory):
    results = []
    print(f"Scanning directory: {directory}")  # Debugging statement
    confidence_files = [f for f in os.listdir(directory) if f.startswith('confidence_') and f.endswith('.json')]
    print(f"Found confidence files: {confidence_files}")  # Debugging statement

    for conf_file in confidence_files:
        model_index = conf_file.split('_')[-1].split('.')[0]  # Extract model index from filename
        base_name = '_'.join(conf_file.split('_')[1:-2])
        confidence_data = json.load(open(os.path.join(directory, conf_file), 'r'))  # Load the JSON data

        # Find the corresponding model file dynamically based on the confidence file name
        base_name = conf_file.replace('confidence_', '').replace(f'_model_{model_index}.json', '')
        model_file = f"{base_name}_model_{model_index}.cif"
        model_path = os.path.join(directory, model_file)
        print(f"Constructed model path: {model_path}")  # Debugging statement

        if os.path.exists(model_path):
            print(f"Model file exists: {model_path}")  # Debugging statement
            results.append({
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
            })
        else:
            print(f"Model file does not exist: {model_path}")  # Debugging statement

    return pd.DataFrame(results)

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

 
# Utility Functions for calculating Centre of Mass and distances.

def centre_of_mass(coordinates):
    # Calculate the 3D position of the centre of mass of set of cartesian coordinates
    result = np.mean(coordinates, axis = 0)
    return result

def calculate_distance_matrix(CA_positions,Centre_of_Mass):
    """
    Funtion to calculate distances between the centre of mass of the ligand and 
    all the 'C-alpha' positions.
    Parameters:
        CA_positions (np.ndarray): Array of C-alpha positions.
        Centre_of_Mass (np.ndarray): Centre of mass position.

    Returns:
        np.ndarray: Distance matrix between the centre of mass and C-alpha positions.
    """
    # Intialize new array:
    distances = []
    for row in CA_positions:
        distances.append(np.linalg.norm(row - Centre_of_Mass))
    return np.array(distances)

def hydrogen_bond_amount(model_path, ligand_asym_unit_id='B', protein_asym_unit_id='A'):
    """
    Function to calculate the number of hydrogen bonds between a ligand and a protein in a CIF model.
    
    Parameters:
    model_path (str): Path to the CIF model file.
    ligand_asym_unit_id (str): Asym unit ID for the ligand.
    protein_asym_unit_id (str): Asym unit ID for the protein.
    
    Returns:
    int: Number of hydrogen bonds between the ligand and the protein.
    """
    structure = strucio.load_structure(model_path)
    ligand_atoms = structure.select(f"asym_id == '{ligand_asym_unit_id}'")
    protein_atoms = structure.select(f"asym_id == '{protein_asym_unit_id}'")
    
    # Calculate hydrogen bonds
    hbonds = biotite.structure.hbond(ligand_atoms, protein_atoms)
    
    return len(hbonds)

