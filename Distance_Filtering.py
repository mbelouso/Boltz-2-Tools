import pandas as pd
import os
import shutil
from filtering_functions import *

"""
Main program loop:
Usage run program in root boltz-2 working directory

Essentially it filters the predictions first, then on the filtered predictions it will
parse all the mmcif data, calculate the mean distance of the ligand to specific residues
then create a new directory and move the filtered cif file to a distance_filtered directory

"""

##### May want to change these variables #####
# 157 is Histadine, 89 is Tyrosine, 211 is Phe
# Note: the distance array starts at 0, so amino acids numbers from chimerax will be +1

binding_site_residue1 = int(157)
binding_site_residue2 = int(89)
binding_site_residue3 = int(211)
distance_threshold = float(10.0)
confidence_threshold = float(0.89)

folder_list = os.listdir('./')

print(f"Found folders: {folder_list}")  # Debugging statement

for folder in folder_list:
    if folder.startswith('boltz_results_'):
        # Process only folders that start with 'boltz_results_'
        # The .json files within this folder are two folders further up in the directory structure
        # so we need to adjust the path accordingly
        # For example, if the folder is 'boltz_results_MIPS-0051357', we will look for
        # 'boltz_results_MIPS-0051357/predictions/MIPS-0051357'
        # Adjust the path to point to the correct directory
        predictions_folder = os.path.join(folder, 'predictions/', folder.split('_')[2])
        if not os.path.exists(predictions_folder):
            print(f"Predictions folder does not exist: {predictions_folder}")
            continue
        else:
            print(f"working on folder: {predictions_folder}")
        print(f"Processing folder: {predictions_folder}")  # Debugging statement
        results_df = parse_boltz2_results(os.path.join('./', predictions_folder))
        # For example, to save it as a CSV file:
        results_df.to_csv(f"{folder}_results.csv", index=False)
    else:
        print(f"Skipping folder: {folder} (does not start with 'boltz_results_')")

# Combine all the CSV files into a single DataFrame

file_list = [f for f in os.listdir('./') if f.endswith('_results.csv')]

combined_df = combine_csv_results(file_list)
combined_df.to_csv('boltz_results_combined.csv', index=False)
print("Combined results saved to 'boltz_results_combined.csv'")  # Debugging statement


# Filtering the DataFrame to include only rows with a confidence score greater than 0.9 
filtered_df = combined_df[combined_df['confidence_score'] > confidence_threshold]
filtered_df.to_csv('boltz_results_filtered.csv', index=False)
print("Filtered results saved to 'boltz_results_filtered.csv'")  # Debugging statement

# Using the filtered DataFrame to copy the corresponding .cif model files into a new directory
output_dir = 'filtered_models'
os.makedirs(output_dir, exist_ok=True)
for index, row in filtered_df.iterrows():
    model_path = row['model_path']
    if os.path.exists(model_path):
        # Copy the model file to the output directory
        os.system(f"cp {model_path} {output_dir}")
        print(f"Copied {model_path} to {output_dir}")
    else:
        print(f"Model file does not exist: {model_path}")  # Debugging statement

# remove the temporary .csv files
for file in file_list:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed temporary file: {file}")  # Debugging statement
    else:
        print(f"File does not exist: {file}")  # Debugging statement


# Copy all the .cif files from the combined results to a new directory
combined_output_dir = 'combined_models'
os.makedirs(combined_output_dir, exist_ok=True)
for file in combined_df['model_path'].unique():
    if os.path.exists(file):
        # Copy the model file to the output directory
        os.system(f"cp {file} {combined_output_dir}")
        print(f"Copied {file} to {combined_output_dir}")
    else:
        print(prediction_id)
        print(f"File does not exist: {file}")  # Debugging statement

# Work on the mmCIF files

distance_filtered_dir = 'distance_filtered'
os.makedirs(distance_filtered_dir, exist_ok=True)

# Initialize a list to store distance matrices and filenames
distance_data = []

for file in os.listdir(output_dir):  # Iterate over files in the output directory
    file_path = os.path.join(output_dir, file)  # Construct the full path to the file
    if not os.path.isfile(file_path):  # Skip if it's not a file
        continue

    try:
        CA_positions = extract_coordinates(file_path, target_atom_id='CA')
        Lig_positions = extract_coordinates(file_path, target_asym_unit_id='B')
        Lig_COM = centre_of_mass(Lig_positions)
        distances = calculate_distance_matrix(CA_positions, Lig_COM)

        # Append the distances and filename to the list
        distance_data.append(pd.Series(distances, name=file))

        if filter_by_sites(distances, binding_site_residue1, binding_site_residue2, binding_site_residue3, distance_threshold)[0]:
            # If the distances pass the filter, copy the file to the distance_filtered_dir
            os.system(f"cp {file_path} {distance_filtered_dir}")
            print(f"Copied {file_path} to {distance_filtered_dir}")
    except FileNotFoundError as e:
        print(f"Error processing file {file_path}: {e}")

# Combine all the distance data into a single DataFrame using pd.concat
distance_df = pd.concat(distance_data, axis=1).transpose()

# Save the DataFrame to a CSV file for later plotting
distance_df.to_csv('distance_matrices.csv', index=True)
print("Distance matrices saved to 'distance_matrices.csv'")

print("/n Boltz-2-Tools Distance Filtering Complete!/n")
print("/n Zipping Results for Sharing!/n")

# Save Data and zip results for sharing:

# Create ZIP archive of the important results for sharing
# Path to the existing archive
archive_name = f'boltz_results_archive_{confidence_threshold}'

# Temporary directory to extract the archive
temp_dir = 'temp_archive'

# Copy the .csv files to the temporary directory
os.makedirs(temp_dir, exist_ok=True)
shutil.copy('boltz_results_combined.csv', temp_dir)
shutil.copy('boltz_results_filtered.csv', temp_dir)
shutil.copy('distance_matrices.csv', temp_dir)

# Copy the distance_filtered directory to the temporary directory
distance_filtered_path = os.path.join(temp_dir, 'distance_filtered')
shutil.copytree('distance_filtered', distance_filtered_path, dirs_exist_ok=True)

# Copy the filtered_models directory to the temporary directory
filtered_models_path = os.path.join(temp_dir, 'filtered_models')
# Copy the combined_models directory to the temporary directoryxist_ok=True)
combined_models_path = os.path.join(temp_dir, 'combined_models')
shutil.copytree('combined_models', combined_models_path, dirs_exist_ok=True)
combined_models_path = os.path.join(temp_dir, 'combined_models')
# Create the archive with the updated contentsdels_path, dirs_exist_ok=True)
shutil.make_archive(archive_name, 'zip', temp_dir)
# Create the archive with the updated contents
# Clean up the temporary directory'zip', temp_dir)
shutil.rmtree(temp_dir)
# Clean up the temporary directory
print("Results Filtered and Collated comrade")