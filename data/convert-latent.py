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

def calculate_wavelength(header, flux):
    crval = header['CRVAL1']
    cdelt = header['CDELT1']
    crpix = header['CRPIX1']
    n_pixels = len(flux)
    index = np.arange(n_pixels)
    return 10 ** (crval + (index - (crpix - 1)) * cdelt)

def create_mask(flux, sigma):
    mask = np.where((flux == 0) | (sigma > 0.02), 0, 1)
    return mask

def get_snr(hdul):
    try:
        snr = hdul[4].data['SNR'][0]
        return snr if snr > 0 else 0
    except KeyError:
        return 0

def convert_fits_to_hdf5(fits_dir, hdf5_path, dataset_name, latent_size, max_files=300, save_interval=10):
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        all_files = [f for f in os.listdir(fits_dir) if f.endswith('.fits')]
        all_files = all_files[:max_files]
        
        index = 0
        
        for i in tqdm(range(0, len(all_files), save_interval), desc="Converting FITS to HDF5"):
            for file_index, file_name in enumerate(all_files[i:i+save_interval]):
                file_path = os.path.join(fits_dir, file_name)
                with fits.open(file_path) as hdul:
                    flux = hdul[1].data.astype(np.float32)
                    header = hdul[1].header
                    wavelength = calculate_wavelength(header, flux).astype(np.float32)
                    snr = get_snr(hdul)
                    sigma = hdul[2].data.astype(np.float32)
                    wavelength_var = calculate_wavelength(header, sigma).astype(np.float32)

                    flux = ensure_native_byteorder(flux)
                    sigma = ensure_native_byteorder(sigma)
                    wavelength = ensure_native_byteorder(wavelength)

                    flux_mask = create_mask(flux, sigma).astype(np.float32)

                    unique_id = f"{dataset_name}_{index}"

                    latent_code = torch.normal(0, 0.01, size=(latent_size,), dtype=torch.float32).numpy()

                    grp = hdf5_file.create_group(file_name)
                    grp.create_dataset('index', data=index)
                    grp.create_dataset('flux', data=flux)
                    grp.create_dataset('wavelength', data=wavelength)
                    grp.create_dataset('snr', data=snr)
                    grp.create_dataset('flux_mask', data=flux_mask)
                    grp.create_dataset('sigma', data=sigma)
                    grp.create_dataset('wavelength_var', data=wavelength_var)
                    grp.create_dataset('unique_id', data=unique_id)  # Add unique ID
                    grp.create_dataset('latent_code', data=latent_code)  # Add latent code

                index += 1
            
            hdf5_file.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FITS files to HDF5 with latent codes")
    parser.add_argument('--fits_dir', type=str, required=True, help="Directory containing FITS files")
    parser.add_argument('--hdf5_path', type=str, required=True, help="Path to output HDF5 file")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--latent_size', type=int, default=10, help="Size of the latent code")
    parser.add_argument('--max_files', type=int, default=1000, help="Maximum number of files to process")
    parser.add_argument('--save_interval', type=int, default=10, help="Interval at which to save progress")

    args = parser.parse_args()

    convert_fits_to_hdf5(
        fits_dir=args.fits_dir,
        hdf5_path=args.hdf5_path,
        dataset_name=args.dataset_name,
        latent_size=args.latent_size,
        max_files=args.max_files,
        save_interval=args.save_interval
    )