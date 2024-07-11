#!/bin/bash


APOGEE_DIR="./test2/apogee"
GALAH_DIR="./test2/galah"
COMBINED_FILE="./test/combined.fits"
HDF5_DIR="./healpixfiles_inter"
LATENT_SIZE=20
CATALOG_PATH='/arc/projects/k-pop/catalogues/galah-dr3v2.fits'
NSIDE=4


# arc/home/Amirabezine/deepSpectra/data/convert_combined.py

python convert_binned.py --apogee_dir "$APOGEE_DIR" --galah_dir "$GALAH_DIR" --combined_file "$COMBINED_FILE" --hdf5_dir "$HDF5_DIR" --latent_size "$LATENT_SIZE" --catalog_path "$CATALOG_PATH" --nside "$NSIDE"


