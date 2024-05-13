import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def filter_wavelengths(wavelengths, flux, lower_bound=15250, upper_bound=15750):
    """
    Filters the wavelengths and corresponding flux data to include only those within a specified range.

    Parameters:
    - wavelengths (np.array): Array of wavelength data.
    - flux (np.array): Corresponding array of flux data.
    - lower_bound (float): Lower wavelength bound.
    - upper_bound (float): Upper wavelength bound.

    Returns:
    - np.array: Filtered wavelengths.
    - np.array: Filtered flux.
    """
    mask = (wavelengths >= lower_bound) & (wavelengths <= upper_bound)
    return wavelengths[mask], flux[mask]



def ensure_native_byteorder(array):
    if array.dtype.byteorder not in ('=', '|'):  # '=' means native, '|' means not applicable
        return array.byteswap().newbyteorder()  # Swap byte order to native
    return array

class filteredAPOGEEDataset(Dataset):
    def __init__(self, directory, max_files=None):
        """
        Args:
            directory (string): Directory with all the FITS files.
            max_files (int): Maximum number of FITS files to load (optional).
        """
        self.data = self.load_data(directory)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.fits')]
        if max_files is not None and max_files < len(all_files):
            self.files = random.sample(all_files, max_files)  # Randomly select max_files from the list
        else:
            self.files = all_files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filepath = self.files[idx]
        with fits.open(filepath) as hdul:
            flux = hdul[1].data.astype(np.float32)
            header = hdul[1].header
            wavelength = self.calculate_wavelength(header, flux).astype(np.float32)
            snr = self.get_snr(hdul)


         
            variation = hdul[2].data.astype(np.float32)
            
            wavelength_var = self.calculate_wavelength(header, variation).astype(np.float32)
            
            resolution = wavelength / (header['CDELT1'] * np.mean(wavelength))




            # Ensure the arrays are in the native byte order
            flux = ensure_native_byteorder(flux)
            variation = ensure_native_byteorder(variation)
            wavelength = ensure_native_byteorder(wavelength)

            # Convert to torch tensors
            flux = torch.from_numpy(flux)
            variation = torch.from_numpy(variation)
            wavelength = torch.from_numpy(wavelength)
            flux_mask = self.create_mask(flux.numpy())  
            flux_mask = torch.from_numpy(flux_mask)

            return  idx, {'wavelength': wavelength,
                    'flux': flux,
                    'snr': snr, 
                       'flux_mask': flux_mask,
                       'variation' : variation,
                   'wavelength_var': wavelength_var}

         
    def calculate_wavelength(self,header, flux):
        """
        Calculates the wavelength array using the FITS header information.
        """
        crval = header['CRVAL1']  # Starting log10 wavelength
        cdelt = header['CDELT1']  # Log10 wavelength increment
        crpix = header['CRPIX1']  # Reference pixel
        n_pixels = len(flux)
        index = np.arange(n_pixels)
        return 10 ** (crval + (index - (crpix - 1)) * cdelt)


    def get_snr(self, hdul):
        try:
            snr = hdul[4].data['SNR'][0]
            return snr if snr > 0 else 0
        except KeyError:
            return 0

    def create_mask(self, flux):
        """
        Creates a mask for the flux array.
        Args:
            flux (ndarray): Array of flux values.
        Returns:
            ndarray: A mask array where the value is 1 if the corresponding flux is zero, and 0 otherwise.
        """
        return np.where(flux == 0, 1, 0)
