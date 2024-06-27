import h5py
import numpy as np
from astropy.io import fits
import os
from tqdm import tqdm
import torch
import argparse

def ensure_native_byteorder(array):
    if array.dtype.byteorder not in ('=', '|'):
        return array.byteswap().newbyteorder()
    return array

def calculate_wavelength_apogee(header, flux):
    crval = header['CRVAL1']
    cdelt = header['CDELT1']
    crpix = header['CRPIX1']
    n_pixels = len(flux)
    index = np.arange(n_pixels)
    return 10 ** (crval + (index - (crpix - 1)) * cdelt)

def calculate_wavelength_galah(header, flux):
    w_start = header['CRVAL1']  # Starting wavelength
    w_delta = header['CDELT1']  # Wavelength increment per pixel
    n_pixels = len(flux)
    wavelength = w_start + w_delta * np.arange(n_pixels) 
    return wavelength

def create_mask_apogee(flux, sigma):
    mask = np.where((flux == 0) | (sigma > 0.02), 0, 1)
    return mask

def create_mask_galah(flux, sigma):
    mask = np.where((flux == 0), 0, 1)
    return mask


def extract_apogee_id(filename):
    prefix = 'aspcapStar-dr17-'
    suffix = '.fits'
    if filename.startswith(prefix) and filename.endswith(suffix):
        return filename[len(prefix):-len(suffix)]
    return None
    
def load_apogee_spectrum(file_path):
    with fits.open(file_path) as hdul:
        flux = hdul[1].data.astype(np.float32)
        header = hdul[1].header
        wavelength = calculate_wavelength_apogee(header, flux).astype(np.float32)
        sigma = hdul[2].data.astype(np.float32)
    
    flux = ensure_native_byteorder(flux)
    sigma = ensure_native_byteorder(sigma)
    wavelength = ensure_native_byteorder(wavelength)
    flux_mask = create_mask_apogee(flux, sigma).astype(np.float32)
    
    return flux, wavelength, sigma, flux_mask

def load_galah_spectrum(file_path):
    with fits.open(file_path) as hdul:
        flux = hdul[4].data.astype(np.float32)
        header = hdul[4].header
        wavelength = calculate_wavelength_galah(header, flux).astype(np.float32)
        sigma = (hdul[1].data * hdul[4].data).astype(np.float32)
    
    flux = ensure_native_byteorder(flux)
    sigma = ensure_native_byteorder(sigma)
    wavelength = ensure_native_byteorder(wavelength)
    flux_mask = create_mask_galah(flux, sigma).astype(np.float32)
    
    return flux, wavelength, sigma, flux_mask

def process_spectra(apogee_dir, galah_dir, apogee_ids, galah_ids, hdf5_file,latent_size):
    index = 0
    processed_apogee = set()
    processed_galah = set()

    for apogee_id, galah_id in tqdm(zip(apogee_ids, galah_ids), total=len(apogee_ids), desc="Converting spectra to HDF5"):
        apogee_file = os.path.join(apogee_dir, f"aspcapStar-dr17-{apogee_id}.fits")
        galah_file = os.path.join(galah_dir, f"{galah_id}.fits")
        apogee_exists = os.path.exists(apogee_file)
        galah_exists = os.path.exists(galah_file)

        if apogee_exists and galah_exists:
            apogee_data = load_apogee_spectrum(apogee_file)
            galah_data = load_galah_spectrum(galah_file)
            
            # Combine data
            flux = np.concatenate((apogee_data[0], galah_data[0]))
            wavelength = np.concatenate((apogee_data[1], galah_data[1]))
            sigma = np.concatenate((apogee_data[2], galah_data[2]))
            flux_mask = np.concatenate((apogee_data[3], galah_data[3]))
            
            unique_id = f"combined_{index}"
            instrument = "combined"
            index += 1
            processed_apogee.add(apogee_id)
            processed_galah.add(galah_id)
        else:
            continue

    #     save_to_hdf5(flux, wavelength, sigma, flux_mask, unique_id, instrument, hdf5_file, index,latent_size)

    # # Process remaining APOGEE files
    # all_apogee_files = {extract_apogee_id(file) for file in os.listdir(apogee_dir) if file.endswith('.fits')}
    # remaining_apogee = all_apogee_files - processed_apogee
    # for apogee_id in remaining_apogee:
    #     apogee_file = os.path.join(apogee_dir, f"aspcapStar-dr17-{apogee_id}.fits")
    #     flux, wavelength, sigma, flux_mask = load_apogee_spectrum(apogee_file)
    #     unique_id = f"apogee_{index}"
    #     instrument = "apogee"
    #     index += 1
    #     save_to_hdf5(flux, wavelength, sigma, flux_mask, unique_id, instrument, hdf5_file, index,latent_size)

    # Process remaining GALAH files
    all_galah_files = {file.split('.')[0] for file in os.listdir(galah_dir)}
    remaining_galah = all_galah_files - processed_galah
    print("Number of remaining GALAH files:", len(remaining_galah))
    print(processed_galah)
    
    for galah_id in remaining_galah:
        galah_file = os.path.join(galah_dir, f"{galah_id}.fits")
        flux, wavelength, sigma, flux_mask = load_galah_spectrum(galah_file)
        unique_id = f"galah_{index}"
        instrument = "galah"
        index += 1
        save_to_hdf5(flux, wavelength, sigma, flux_mask, unique_id, instrument, hdf5_file, index,latent_size)

def save_to_hdf5(flux, wavelength, sigma, flux_mask, unique_id, instrument, hdf5_file, index,latent_size):
    latent_code = torch.normal(0,1, size=(latent_size,), dtype=torch.float32).numpy()
    grp = hdf5_file.create_group(unique_id)
    grp.create_dataset('flux', data=flux, compression="gzip", compression_opts=9)
    grp.create_dataset('wavelength', data=wavelength, compression="gzip", compression_opts=9)
    grp.create_dataset('sigma', data=sigma, compression="gzip", compression_opts=9)
    grp.create_dataset('flux_mask', data=flux_mask)
    grp.create_dataset('latent_code', data=latent_code)
    grp.create_dataset('instrument', data=instrument)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert APOGEE and GALAH FITS files to combined HDF5")
    parser.add_argument('--apogee_dir', type=str, required=True, help="Directory containing APOGEE FITS files")
    parser.add_argument('--galah_dir', type=str, required=True, help="Directory containing GALAH FITS files")
    parser.add_argument('--combined_file', type=str, required=True, help="Path to combined FITS catalog")
    parser.add_argument('--hdf5_path', type=str, required=True, help="Path to output HDF5 file")
    parser.add_argument('--latent_size', type=int, default=10, help="Size of the latent code")

    args = parser.parse_args()
    with h5py.File(args.hdf5_path, 'w') as hdf5_file, fits.open(args.combined_file) as hdul:
        data = hdul[1].data
        apogee_ids = data['APOGEE_ID']
        galah_ids = data['sobject_id']
        process_spectra(args.apogee_dir, args.galah_dir, apogee_ids, galah_ids, hdf5_file, args.latent_size)
