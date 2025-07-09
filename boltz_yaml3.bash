#!/bin/bash

# For this to run, you will need to have the *.yaml files in a directory called 'yaml' in the same directory as this script.


#SBATCH --job-name=boltz_abinit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --mem=100GB 
#SBATCH --time=72:00:00

# GPU Request
#SBATCH --gres=gpu:1

# Set the file for output (stdout)
#SBATCH --output=Boltz-%j.out

# Set the file for error log (stderr)
#SBATCH --error=Boltz-%j.err

# Initialize Conda Environment

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/boltz
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

# Check if the 'yaml' directory exists
if [ ! -d "yaml3" ]; then
    echo "Directory 'yaml3' does not exist. Please create it and place your YAML files there."
    exit 1
fi

# Run the Boltzmann Ab Initio script
for yaml_file in yaml3/*.yaml; do
    if [ -f "$yaml_file" ]; then
        echo "Processing $yaml_file..."
        boltz predict "$yaml_file" --use_msa_server --preprocessing-threads 16
    fi
done
echo "All calculations completed comrade"
