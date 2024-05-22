import cupy as cp
import h5py
from astropy.io import fits
import os
from tqdm import tqdm

def ensure_native_byteorder(array):
    if array.dtype.byteorder not in ('=', '|'):  # '=' means native, '|' means not applicable
        return array.byteswap().newbyteorder()  # Swap byte order to native
    return array

def calculate_wavelength(header, flux):
    """
    Calculates the wavelength array using the FITS header information.
    """
    crval = header['CRVAL1']  # Starting log10 wavelength
    cdelt = header['CDELT1']  # Log10 wavelength increment
    crpix = header['CRPIX1']  # Reference pixel
    n_pixels = len(flux)
    index = cp.arange(n_pixels)
    return 10 ** (crval + (index - (crpix - 1)) * cdelt)

def create_mask(flux, sigma):
    """
    Creates a mask for the flux array where the mask is 0 if the flux is zero or sigma > 0.5, and 1 otherwise.
    """
    mask = cp.where((flux == 0) | (sigma > 0.5), 0, 1)
    return mask

def get_snr(hdul):
    try:
        snr = hdul[4].data['SNR'][0]
        return snr if snr > 0 else 0
    except KeyError:
        return 0

def convert_fits_to_hdf5(fits_dir, hdf5_path, max_files=300, save_interval=10):
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        all_files = [f for f in os.listdir(fits_dir) if f.endswith('.fits')]
        all_files = all_files[:max_files]  # Limit the number of files

        for i in tqdm(range(0, len(all_files), save_interval), desc="Converting FITS to HDF5"):
            for file_name in all_files[i:i+save_interval]:
                file_path = os.path.join(fits_dir, file_name)
                with fits.open(file_path) as hdul:
                    flux = cp.array(hdul[1].data.astype(cp.float32))
                    header = hdul[1].header
                    wavelength = calculate_wavelength(header, flux).astype(cp.float32)
                    snr = get_snr(hdul)
                    sigma = cp.array(hdul[2].data.astype(cp.float32))
                    wavelength_var = calculate_wavelength(header, sigma).astype(cp.float32)

                    flux = ensure_native_byteorder(flux)
                    sigma = ensure_native_byteorder(sigma)
                    wavelength = ensure_native_byteorder(wavelength)

                    flux_mask = create_mask(flux, sigma).astype(cp.float32)

                    grp = hdf5_file.create_group(file_name)
                    grp.create_dataset('flux', data=cp.asnumpy(flux))
                    grp.create_dataset('wavelength', data=cp.asnumpy(wavelength))
                    grp.create_dataset('snr', data=snr)
                    grp.create_dataset('flux_mask', data=cp.asnumpy(flux_mask))
                    grp.create_dataset('sigma', data=cp.asnumpy(sigma))
                    grp.create_dataset('wavelength_var', data=cp.asnumpy(wavelength_var))
            
            hdf5_file.flush()  # Ensure data is written to disk

if __name__ == "__main__":
    convert_fits_to_hdf5(
        "../../../../projects/k-pop/spectra/apogee/dr17", 
        "../data/hdf5/spectra.hdf5", 
        max_files=50000
    )
