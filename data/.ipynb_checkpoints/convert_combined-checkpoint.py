import h5py
import numpy as np
from astropy.io import fits
import os
from tqdm import tqdm
import torch
import argparse
import healpy as hp


def get_healpix_index(ra, dec, nside):
    theta = np.radians(90 - dec)
    phi = np.radians(ra)
    return hp.ang2pix(nside, theta, phi)


def load_galah_catalog(catalog_path):
    with fits.open(catalog_path) as hdul:
        data = hdul[1].data
    return {row['sobject_id']: row for row in data}
    
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



def extract_apogee_metadata(file_path):
    with fits.open(file_path) as hdul:
        # Access the table data in the 4th HDU
        data = hdul[4].data
        # Extract metadata, using np.nan as default for missing values
        return {
            'temp': data['RV_TEFF'][0] if 'RV_TEFF' in hdul[4].columns.names else np.nan,
            'temp_err': data['RV_TEFF_ERR'][0] if 'RV_TEFF_ERR' in hdul[4].columns.names else np.nan,
            'logg': data['RV_LOGG'][0] if 'RV_LOGG' in hdul[4].columns.names else np.nan,
            'logg_err': data['RV_LOGG_ERR'][0] if 'RV_LOGG_ERR' in hdul[4].columns.names else np.nan,
            'fe_h': data['RV_FEH'][0] if 'RV_FEH' in hdul[4].columns.names else np.nan,
            'fe_h_err': data['RV_FEH_ERR'][0] if 'RV_FEH_ERR' in hdul[4].columns.names else np.nan,
            'rv': data['VHELIO_AVG'][0] if 'VHELIO_AVG' in hdul[4].columns.names else np.nan,
            'rv_err': data['VERR'][0] if 'VERR' in hdul[4].columns.names else np.nan,
            'ra': data['RA'][0],  # RA is expected to always be present
            'dec': data['DEC'][0]  # DEC is expected to always be present
        }


def extract_galah_metadata(galah_catalog, sobject_id):
    print(sobject_id)
    if sobject_id in galah_catalog:
        row = galah_catalog[sobject_id]
        return {
            'temp': row['teff'],
            'temp_err': row['e_teff'],
            'logg': row['logg'],
            'logg_err': row['e_logg'],
            'fe_h': row['fe_h'],
            'fe_h_err': row['e_fe_h'],
            'rv': row['rv_galah'],
            'rv_err': row['e_rv_galah'],
            'ra': row['ra_dr2'],
            'dec': row ['dec_dr2']
        }
    return None
    
def save_to_hdf5(flux, wavelength, sigma, flux_mask, unique_id, instrument_type, hdf5_dir, index, latent_size, combined_meta, healpix_index):
    hdf5_path = os.path.join(hdf5_dir, f"spectra_healpix_{healpix_index}.hdf5")
    latent_code = torch.normal(0, 1, size=(latent_size,), dtype=torch.float32).numpy()
    
    with h5py.File(hdf5_path, 'a') as hdf5_file:
        # Check if the group already exists, if not create it
        if unique_id in hdf5_file:
            grp = hdf5_file[unique_id]
        else:
            grp = hdf5_file.create_group(unique_id)

        grp.create_dataset('flux', data=flux, compression="gzip", compression_opts=9)
        grp.create_dataset('wavelength', data=wavelength, compression="gzip", compression_opts=9)
        grp.create_dataset('sigma', data=sigma, compression="gzip", compression_opts=9)
        grp.create_dataset('flux_mask', data=flux_mask)
        grp.create_dataset('latent_code', data=latent_code)
        grp.create_dataset('instrument_type', data=np.string_(instrument_type))
        grp.create_dataset('unique_id', data=np.string_(unique_id))
        
        # Handle metadata
        for key, value in combined_meta.items():
            try:
                if value is None:
                    # Assuming all None values can be stored as np.nan in the dataset
                    grp.create_dataset(key, data=np.array([np.nan], dtype=np.float32))
                elif isinstance(value, list):
                    # Convert None to np.nan for float compatibility in HDF5
                    clean_value = [np.nan if v is None else v for v in value]
                    grp.create_dataset(key, data=np.array(clean_value, dtype=np.float32))
                elif isinstance(value, (int, float)):
                    # Single numeric values
                    grp.create_dataset(key, data=np.float32(value))
                elif isinstance(value, str):
                    # Single string values
                    grp.create_dataset(key, data=np.string_(value))
                elif isinstance(value, np.ndarray):
                    # Directly handle numpy arrays (for instruments)
                    grp.create_dataset(key, data=value, dtype=value.dtype)
                else:
                    print(f"Unsupported data type for key {key}: {type(value)}")
            except Exception as e:
                print(f"Error while saving {key}: {str(e)}")



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
        instruments = np.array([0, 1, 2], dtype=np.int32)
    
    flux = ensure_native_byteorder(flux)
    sigma = ensure_native_byteorder(sigma)
    wavelength = ensure_native_byteorder(wavelength)
    flux_mask = create_mask_apogee(flux, sigma).astype(np.float32)

    return flux, wavelength, sigma, flux_mask, instruments



def load_galah_spectrum(galah_dir, galah_id):
    fluxes = []
    wavelengths = []
    sigmas = []
    flux_masks = []
    instruments = []

    # print("I am in load galah spectrum, my galah id is " , galah_id)
    
    for i in range (1,5):
        file_path = os.path.join(galah_dir, f"{galah_id}{i}.fits")
        

        if not os.path.exists(file_path):
            print(f"Warning: File not found for {file_path}")
            continue

        instruments = np.append(instruments , i+2)
        with fits.open(file_path) as hdul:
            flux = hdul[4].data.astype(np.float32)
            header = hdul[4].header
            wavelength = calculate_wavelength_galah(header, flux).astype(np.float32)
            sigma = (hdul[1].data * hdul[4].data).astype(np.float32)
        
        flux = ensure_native_byteorder(flux)
        sigma = ensure_native_byteorder(sigma)
        wavelength = ensure_native_byteorder(wavelength)
        flux_mask = create_mask_galah(flux, sigma).astype(np.float32)
        
        fluxes.append(flux)
        wavelengths.append(wavelength)
        sigmas.append(sigma)
        flux_masks.append(flux_mask)

    if not fluxes:
        raise ValueError(f"No valid GALAH files found for {galah_id}")

    return (np.concatenate(fluxes), 
            np.concatenate(wavelengths), 
            np.concatenate(sigmas), 
            np.concatenate(flux_masks), 
            instruments)



def process_spectra(apogee_dir, galah_dir, apogee_ids, galah_ids, hdf5_dir, latent_size, catalog_path, nside):
    index = 0
    processed_apogee = set()
    processed_galah = set()
    galah_catalog = load_galah_catalog(catalog_path)
    for i, (apogee_id, galah_id) in enumerate(tqdm(zip(apogee_ids, galah_ids), total=len(apogee_ids), desc="Converting spectra to HDF5")):
        apogee_file = os.path.join(apogee_dir, f"aspcapStar-dr17-{apogee_id}.fits")
        apogee_exists = os.path.exists(apogee_file)

        try:
            if apogee_exists:
                apogee_data = load_apogee_spectrum(apogee_file)
                apogee_meta = extract_apogee_metadata(apogee_file)
            else:
                apogee_data = None
                apogee_meta = None

            # print("looking for galah id: " , galah_id)
            galah_data = load_galah_spectrum(galah_dir, galah_id)
            galah_meta = extract_galah_metadata(galah_catalog, galah_id)

            if galah_meta is None:
                print(f"Warning: No GALAH metadata found for sobject_id {galah_id}")
                continue

            if apogee_exists:
                # Combine spectral data
                flux = np.concatenate((apogee_data[0], galah_data[0]))
                wavelength = np.concatenate((apogee_data[1], galah_data[1]))
                sigma = np.concatenate((apogee_data[2], galah_data[2]))
                flux_mask = np.concatenate((apogee_data[3], galah_data[3]))
                instruments = np.concatenate((apogee_data[4], galah_data[4]))  
                instrument_type = "combined"
            else:
                flux, wavelength, sigma, flux_mask, instruments = galah_data
                instrument_type = "galah"

            combined_meta = {
                'temp': [apogee_meta['temp'] if apogee_meta else None, galah_meta['temp']],
                'temp_err': [apogee_meta['temp_err'] if apogee_meta else None, galah_meta['temp_err']],
                'logg': [apogee_meta['logg'] if apogee_meta else None, galah_meta['logg']],
                'logg_err': [apogee_meta['logg_err'] if apogee_meta else None, galah_meta['logg_err']],
                'rv': [apogee_meta['rv'] if apogee_meta else None, galah_meta['rv']],
                'rv_err': [apogee_meta['rv_err'] if apogee_meta else None, galah_meta['rv_err']],
                'ra': [apogee_meta['ra'] if apogee_meta else None, galah_meta['ra']],
                'dec': [apogee_meta['dec'] if apogee_meta else None, galah_meta['dec']],
                'instruments': instruments,
                'obj_class' : "Star"
            }

            ra = combined_meta['ra'][0] if combined_meta['ra'][0] is not None else combined_meta['ra'][1]
            dec = combined_meta['dec'][0] if combined_meta['dec'][0] is not None else combined_meta['dec'][1]
            healpix_index = get_healpix_index(ra, dec, nside)

            unique_id = f"{healpix_index}_{instrument_type}_{index}"
            index += 1

            if apogee_exists:
                processed_apogee.add(apogee_id)
            processed_galah.add(galah_id)

        except ValueError as e:
            print(f"Skipping {galah_id}: {str(e)}")
            continue

       
        save_to_hdf5(flux, wavelength, sigma, flux_mask, unique_id, instrument_type, hdf5_dir, index, latent_size, combined_meta, healpix_index)
 

        

    # Process remaining APOGEE files
    all_apogee_files = {extract_apogee_id(file) for file in os.listdir(apogee_dir) if file.endswith('.fits')}
    remaining_apogee = all_apogee_files - processed_apogee

    for apogee_id in remaining_apogee:
        apogee_file = os.path.join(apogee_dir, f"aspcapStar-dr17-{apogee_id}.fits")
        print("apogee file " , apogee_file)
        flux, wavelength, sigma, flux_mask, instruments = load_apogee_spectrum(apogee_file)
        
        instrument_type = "apogee"
        index += 1

        apogee_meta = extract_apogee_metadata(apogee_file)
        combined_meta = {
            
            'temp': [apogee_meta['temp'], None],
            'temp_err': [apogee_meta['temp_err'], None],
            'logg': [apogee_meta['logg'], None],
            'logg_err': [apogee_meta['logg_err'], None],
            'fe_h': [apogee_meta['fe_h'], None],
            'fe_h_err': [apogee_meta['fe_h_err'], None],
            'rv': [apogee_meta['rv'], None],
            'rv_err': [apogee_meta['rv_err'], None],

            'ra': [apogee_meta['ra'], None],
            'dec': [apogee_meta['dec'], None],
            'instruments': instruments,
            'obj_class' : "Star"
            
        }

        ra = combined_meta['ra'][0] 
        dec = combined_meta['dec'][0] 
        healpix_index = get_healpix_index(ra, dec, nside)
        
        unique_id = f"{healpix_index}_{instrument_type}_{index}"
        
        save_to_hdf5(flux, wavelength, sigma, flux_mask, unique_id, instrument_type, hdf5_dir, index, latent_size, combined_meta, healpix_index)
        
    print("finished APOGEE")


    
    # Process remaining GALAH files
    all_galah_files = {int(file[:15]) for file in os.listdir(galah_dir) if file.endswith('.fits')}
    remaining_galah = all_galah_files - processed_galah
    # print("all_galah_files = " ,all_galah_files)
    # print("----------------------------------------")
    # print("remaining_galah= ", remaining_galah)
    # print("----------------------------------------")
    # print("processed_galah", processed_galah)
    # print("----------------------------------------")
    for galah_id in remaining_galah:
        try:
            # print("looking for galah id remaining: " , galah_id)
            flux, wavelength, sigma, flux_mask, instruments = load_galah_spectrum(galah_dir, galah_id)
            
            instrument_type = "galah"
            index += 1
            print("working on  galah : " , galah_id)
            galah_meta = extract_galah_metadata(galah_catalog, galah_id)
            if galah_meta is None:
                print(f"Warning: No GALAH metadata found for sobject_id {galah_id}")
                continue

            combined_meta = {
                'temp': [None, galah_meta['temp']],
                'temp_err': [None, galah_meta['temp_err']],
                'logg': [None, galah_meta['logg']],
                'logg_err': [None, galah_meta['logg_err']],
                'fe_h': [None, galah_meta['fe_h']],
                'fe_h_err': [None, galah_meta['fe_h_err']],
                'rv': [None, galah_meta['rv']],
                'rv_err': [None, galah_meta['rv_err']],
                'ra': [None, galah_meta['ra']],
                'dec': [None, galah_meta['dec']],
                'instruments': instruments,
                'obj_class' : "Star"
            }

            ra = combined_meta['ra'][1] 
            dec = combined_meta['dec'][1] 
            healpix_index = get_healpix_index(ra, dec, nside)
            unique_id = f"{healpix_index}_{instrument_type}_{index}"
        
            save_to_hdf5(flux, wavelength, sigma, flux_mask, unique_id, instrument_type, hdf5_dir, index, latent_size, combined_meta, healpix_index)

        except ValueError as e:
            print(f"Skipping {galah_id}: {str(e)}")
            continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert APOGEE and GALAH FITS files to combined HDF5")
    parser.add_argument('--apogee_dir', type=str, required=True, help="Directory containing APOGEE FITS files")
    parser.add_argument('--galah_dir', type=str, required=True, help="Directory containing GALAH FITS files")
    parser.add_argument('--combined_file', type=str, required=True, help="Path to combined FITS catalog")
    # parser.add_argument('--hdf5_path', type=str, required=True, help="Path to output HDF5 file")
    parser.add_argument('--latent_size', type=int, default=10, help="Size of the latent code")
    parser.add_argument('--catalog_path', type=str, required=True, help="Path to Galah catalog fits file")
    parser.add_argument('--hdf5_dir', type=str, required=True, help="Directory to output HDF5 files")
    parser.add_argument('--nside', type=int, default=32, help="HEALPix NSIDE parameter")

    args = parser.parse_args()
    
    with fits.open(args.combined_file) as hdul:
        apogee_ids = hdul[1].data['APOGEE_ID']
        galah_ids = hdul[1].data['sobject_id'] 

        
        process_spectra(args.apogee_dir, args.galah_dir, apogee_ids, galah_ids, args.hdf5_dir, args.latent_size, args.catalog_path, args.nside)

        # process_spectra(args.apogee_dir, args.galah_dir, apogee_ids, galah_ids, hdf5_file, args.latent_size , args.catalog_path)
