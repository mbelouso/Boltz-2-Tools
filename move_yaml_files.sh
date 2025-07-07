#!/bin/bash

# Script to move YAML files to 4 separate directories for parallel processing
# Usage: ./move_yaml_files.sh  
# Ensure the script is run from the directory containing the YAML files

# Create directories if they do not exist
mkdir -p yaml1 yaml2 yaml3 yaml4

# Check if any YAML files exist
if ! ls *.yaml 1> /dev/null 2>&1; then
    echo "No YAML files found in the current directory."
    exit 1
fi

# Run a for loop where each iteration moves a YAML into one of the directories then cycles through
counter=1
for file in *.yaml; do
    # Double-check the file exists (in case it was moved/deleted during iteration)
    if [ -f "$file" ]; then
        case $counter in
            1) mv "$file" yaml1/ && echo "Moved $file to yaml1/" ;;
            2) mv "$file" yaml2/ && echo "Moved $file to yaml2/" ;;
            3) mv "$file" yaml3/ && echo "Moved $file to yaml3/" ;;
            4) mv "$file" yaml4/ && echo "Moved $file to yaml4/" ;;
        esac
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to move $file"
            exit 1
        fi
        
        counter=$((counter % 4 + 1))  # Cycle through 1 to 4
    fi
done

# Print a message indicating completion
echo "YAML files have been moved to yaml1, yaml2, yaml3, and yaml4 directories."
echo "You can now run the processing in parallel using these directories."