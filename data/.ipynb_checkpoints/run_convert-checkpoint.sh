#!/bin/bash


FITS_DIR="../../../../projects/k-pop/spectra/apogee/dr17"
HDF5_PATH="../data/hdf5/apogee4000.hdf5"
DATASET_NAME="apogee_dr17"
LATENT_SIZE=20
MAX_FILES=4000



python convert-latent.py --fits_dir "$FITS_DIR" --hdf5_path "$HDF5_PATH" --dataset_name "$DATASET_NAME" --latent_size "$LATENT_SIZE" --max_files "$MAX_FILES"
