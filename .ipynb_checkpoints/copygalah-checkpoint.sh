#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="/arc/projects/k-pop/spectra/galah/dr3"
DEST_DIR="./data/test2/galah"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Read from the galahfiles.txt file
while IFS= read -r galah_id; do
    # Loop over each possible file ending
    for i in 1 2 3 4; do
        file_name="${galah_id}${i}.fits"
        # Check if the file exists and copy it
        if [ -f "$SOURCE_DIR/$file_name" ]; then
            cp "$SOURCE_DIR/$file_name" "$DEST_DIR"
            echo "Copied $file_name to $DEST_DIR"
        else
            echo "$file_name does not exist."
        fi
    done
done < "./galahfiles.txt"
