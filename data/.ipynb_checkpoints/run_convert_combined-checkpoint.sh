#!/bin/bash


APOGEE_DIR="./test/apogee"
GALAH_DIR="./test/galah"
COMBINED_FILE="./test/combined.fits"
HDF5_PATH="./test/galahtest.hdf5"
LATENT_SIZE=20


# arc/home/Amirabezine/deepSpectra/data/convert_combined.py

python convert_combined.py --apogee_dir "$APOGEE_DIR" --galah_dir "$GALAH_DIR" --combined_file "$COMBINED_FILE" --hdf5_path "$HDF5_PATH" --latent_size "$LATENT_SIZE"


