from pymol import cmd, stored
import os

# Need to write a loop that goes through all the .cif files loaded in PyMOL
# and performs a cealign on each of them to a common reference structure.
# The reference structure is the first loaded structure in PyMOL.

# Get the first loaded structure as the reference
reference_structure = cmd.get_names('all')[0] if cmd.get_names('all') else None
# Loop through all loaded structures
if reference_structure:
    for file in cmd.get_names('all'):
        if file != reference_structure:  # Skip the reference structure itself
            cmd.super(file, reference_structure)
            print(f"Aligned {file} to {reference_structure}")
else:
    print("No structures loaded in PyMOL to align.")   

# Save the aligned structures
output_dir = 'aligned_structures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Save all aligned structures to a PDB file
cmd.save(f"{output_dir}/aligned_structures.cif", 'all')
