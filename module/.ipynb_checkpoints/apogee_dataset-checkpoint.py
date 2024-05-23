import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def ensure_native_byteorder(array):
    if array.dtype.byteorder not in ('=', '|'):  # '=' means native, '|' means not applicable
        return array.byteswap().newbyteorder()  # Swap byte order to native
    return array

class APOGEEDataset(Dataset):
    def __init__(self, directory, max_files=None):
        """
        Args:
            directory (string): Directory with all the FITS files.
            max_files (int): Maximum number of FITS files to load (optional).
        """

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


            sigma = hdul[2].data.astype(np.float32)
            
            wavelength_var = self.calculate_wavelength(header, sigma).astype(np.float32)
    
            resolution = wavelength / (header['CDELT1'] * np.mean(wavelength))




            # Ensure the arrays are in the native byte order
            flux = ensure_native_byteorder(flux)
            sigma = ensure_native_byteorder(sigma)
            wavelength = ensure_native_byteorder(wavelength)
         

            # Convert to torch tensors
            flux = torch.from_numpy(flux)
            sigma = torch.from_numpy(sigma)
            wavelength = torch.from_numpy(wavelength)

            flux_mask = self.create_mask(flux.numpy(), sigma.numpy())  
            flux_mask = torch.from_numpy(flux_mask)

            return  idx, {'wavelength': wavelength,
                    'flux': flux,
                    'snr': snr, 
                    'flux_mask': flux_mask,
                  
                   'sigma' : sigma,
                         
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

    def create_mask(self, flux, sigma):
        """
        Creates a mask for the flux array where the mask is 0 if the flux is zero or sigma > 0.5, and 1 otherwise.
        
        Args:
            flux (ndarray): Array of flux values.
            sigma (ndarray): Array of sigma values corresponding to each flux value.
        
        Returns:
            ndarray: A mask array where the value is 0 if the corresponding flux is zero or sigma > 0.5, and 1 otherwise.
        """
        mask = np.where((flux == 0) | (sigma > 0.5), 0, 1)
        return mask

