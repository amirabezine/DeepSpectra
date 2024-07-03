#!/bin/bash


APOGEE_DIR="./test2/apogee"
GALAH_DIR="./test2/galah"
COMBINED_FILE="./test/combined.fits"
HDF5_PATH="./test2/data.hdf5"
LATENT_SIZE=20
CATALOG_PATH='/arc/projects/k-pop/catalogues/galah-dr3v2.fits'

# arc/home/Amirabezine/deepSpectra/data/convert_combined.py

python convert_combined.py --apogee_dir "$APOGEE_DIR" --galah_dir "$GALAH_DIR" --combined_file "$COMBINED_FILE" --hdf5_path "$HDF5_PATH" --latent_size "$LATENT_SIZE" --catalog_path "$CATALOG_PATH"


