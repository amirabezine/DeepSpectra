import torch
from torch.utils.data import IterableDataset, DataLoader
import h5py
import numpy as np
import random
import os
from glob import glob
from torch.nn.utils.rnn import pad_sequence

def ensure_native_byteorder(array):
    if array.dtype.byteorder not in ('=', '|'):
        array = array.byteswap().new_byteorder('=')
    return array

class IterableSpectraDataset(IterableDataset):
    def __init__(self, hdf5_dir, n_samples_per_spectrum=500, n_subspectra=10, validation_split=0.2, is_validation=False, max_files=None, yield_full_spectrum=False):
        self.hdf5_dir = hdf5_dir
        self.n_samples_per_spectrum = n_samples_per_spectrum
        self.n_subspectra = n_subspectra
        self.validation_split = validation_split
        self.is_validation = is_validation
        self.file_list = glob(os.path.join(hdf5_dir, 'spectra_healpix_*.hdf5'))
        self.yield_full_spectrum = yield_full_spectrum

        random.shuffle(self.file_list)
        split_idx = int(len(self.file_list) * (1 - validation_split))
        self.file_list = self.file_list[split_idx:] if is_validation else self.file_list[:split_idx]

        if max_files is not None:
            self.file_list = self.file_list[:max_files]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            per_worker = int(np.ceil(len(self.file_list) / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_list))

        for file_path in self.file_list[iter_start:iter_end]:
            if self.yield_full_spectrum:
                yield from self.process_file_full_spectrum(file_path)
            else:
                yield from self.process_file(file_path)

    def process_file(self, file_path):
        with h5py.File(file_path, 'r') as f:
            healpix_index = os.path.basename(file_path).split('_')[2].split('.')[0]
            for group_name in f.keys():
                group_healpix_index = group_name.split('_')[0]
                if group_healpix_index == healpix_index:
                    group = f[group_name]
                    
                    try:
                        unique_id = group['unique_id'][()].decode('utf-8')
                        flux = ensure_native_byteorder(group['flux'][:])
                        sigma = ensure_native_byteorder(group['sigma'][:])
                        wavelength = ensure_native_byteorder(group['wavelength'][:])
                        flux_mask = ensure_native_byteorder(group['flux_mask'][:])
                        latent_code = ensure_native_byteorder(group['latent_code'][:])

                        metadata = {
                            'dec': group['dec'][()],
                            'instrument_type': group['instrument_type'][()].decode('utf-8'),
                            'instruments': group['instruments'][()],
                            'logg': group['logg'][()],
                            'logg_err': group['logg_err'][()],
                            'obj_class': group['obj_class'][()].decode('utf-8'),
                            'ra': group['ra'][()],
                            'rv': group['rv'][()],
                            'rv_err': group['rv_err'][()],
                            'temp': group['temp'][()],
                            'temp_err': group['temp_err'][()],
                        }

                        for _ in range(self.n_subspectra):
                            indices = np.random.choice(len(wavelength), self.n_samples_per_spectrum, replace=False)

                            flux_tensor = torch.tensor(flux[indices], dtype=torch.float32)
                            sigma_tensor = torch.tensor(sigma[indices], dtype=torch.float32)
                            mask_tensor = torch.tensor(flux_mask[indices], dtype=torch.float32)
                            wavelength_tensor = torch.tensor(wavelength[indices], dtype=torch.float32)
                            latent_code_tensor = torch.tensor(latent_code, dtype=torch.float32)
                            
                            yield unique_id, flux_tensor, sigma_tensor, mask_tensor, wavelength_tensor, latent_code_tensor, metadata
                    except KeyError as e:
                        print(f"KeyError: {e} in group {group_name}")
                    except Exception as e:
                        print(f"Exception: {e} in group {group_name}")

    def process_file_full_spectrum(self, file_path):
        with h5py.File(file_path, 'r') as f:
            healpix_index = os.path.basename(file_path).split('_')[2].split('.')[0]
            for group_name in f.keys():
                group_healpix_index = group_name.split('_')[0]
                if group_healpix_index == healpix_index:
                    group = f[group_name]
                    
                    try:
                        unique_id = group['unique_id'][()].decode('utf-8')
                        flux = ensure_native_byteorder(group['flux'][:])
                        sigma = ensure_native_byteorder(group['sigma'][:])
                        wavelength = ensure_native_byteorder(group['wavelength'][:])
                        flux_mask = ensure_native_byteorder(group['flux_mask'][:])
                        latent_code = ensure_native_byteorder(group['latent_code'][:])

                        metadata = {
                            'dec': group['dec'][()],
                            'instrument_type': group['instrument_type'][()].decode('utf-8'),
                            'instruments': group['instruments'][()],
                            'logg': group['logg'][()],
                            'logg_err': group['logg_err'][()],
                            'obj_class': group['obj_class'][()].decode('utf-8'),
                            'ra': group['ra'][()],
                            'rv': group['rv'][()],
                            'rv_err': group['rv_err'][()],
                            'temp': group['temp'][()],
                            'temp_err': group['temp_err'][()],
                        }

                        flux_tensor = torch.tensor(flux, dtype=torch.float32)
                        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
                        mask_tensor = torch.tensor(flux_mask, dtype=torch.float32)
                        wavelength_tensor = torch.tensor(wavelength, dtype=torch.float32)
                        latent_code_tensor = torch.tensor(latent_code, dtype=torch.float32)
                        
                        yield unique_id, flux_tensor, sigma_tensor, mask_tensor, wavelength_tensor, latent_code_tensor, metadata

                    except KeyError as e:
                        print(f"KeyError: {e} in group {group_name}")
                    except Exception as e:
                        print(f"Exception: {e} in group {group_name}")

    def load_latent_vectors(self, loaders, latent_dim, device):
        latent_vectors = []
        dict_latent_codes = {}
        processed_ids = set()
    
        try:
            idx = 0
            for file_path in self.file_list:
                healpix_index = os.path.basename(file_path).split('_')[2].split('.')[0]
                with h5py.File(file_path, 'r') as hdf5_file:
                    for group_name in hdf5_file.keys():
                        unique_id = group_name
                        unique_id_healpix_index = unique_id.split('_')[0]
                        if unique_id_healpix_index == healpix_index and unique_id not in processed_ids:
                            processed_ids.add(unique_id)
                            try:
                                group = hdf5_file[unique_id]
                                if 'latent_code' in group:
                                    dict_latent_codes[unique_id] = idx
                                    data = torch.tensor(group['latent_code'][()], dtype=torch.float32, device=device)
                                    latent_vectors.append(data)
                                else:
                                    print(f"latent_code not found for unique_id {unique_id}")
                                    latent_vectors.append(torch.zeros(latent_dim, device=device))
                                idx += 1
                            except KeyError as e:
                                print(f"KeyError: {e} - unique_id {unique_id} not found in file {file_path}")
                                latent_vectors.append(torch.zeros(latent_dim, device=device))
                                idx += 1
        except OSError as e:
            print(f"Failed to open file {self.hdf5_dir}: {e}")
    
        if not latent_vectors:
            raise RuntimeError("No latent vectors were loaded. Check the data directory and file contents.")
            
        latent_codes = torch.stack(latent_vectors, dim=0).requires_grad_(True)
        return latent_codes, dict_latent_codes

    def save_latent_vectors_to_hdf5(self, dict_latent_codes, latent_vectors, epoch):
        try:
            for file_path in self.file_list:
                healpix_index = os.path.basename(file_path).split('_')[2].split('.')[0]
                with h5py.File(file_path, 'a') as hdf5_file:
                    print(f"Saving latent vectors to file: {file_path} for epoch {epoch}")
                    for unique_id, index in dict_latent_codes.items():
                        unique_id_healpix_index = unique_id.split('_')[0]
                        if unique_id_healpix_index == healpix_index:
                            try:
                                group = hdf5_file[unique_id]
                                versioned_key = f"optimized_latent_code/epoch_{epoch}"
                                if versioned_key in group:
                                    del group[versioned_key]
                                group.create_dataset(versioned_key, data=latent_vectors[index].cpu().detach().numpy())
                            except KeyError as e:
                                print(f"KeyError: {e} - unique_id {unique_id} not found in file {file_path}")
        except OSError as e:
            print(f"Failed to open file {self.hdf5_dir}: {e}")

    def save_last_latent_vectors_to_hdf5(self, dict_latent_codes, latent_vectors):
        try:
            for file_path in self.file_list:
                healpix_index = os.path.basename(file_path).split('_')[2].split('.')[0]
                with h5py.File(file_path, 'a') as hdf5_file:
                    print(f"Saving last latent vectors to file: {file_path}")
                    for unique_id, index in dict_latent_codes.items():
                        unique_id_healpix_index = unique_id.split('_')[0]
                        if unique_id_healpix_index == healpix_index:
                            try:
                                group = hdf5_file[unique_id]
                                versioned_key = "optimized_latent_code/latest"
                                if versioned_key in group:
                                    del group[versioned_key]
                                group.create_dataset(versioned_key, data=latent_vectors[index].cpu().detach().numpy())
                            except KeyError as e:
                                print(f"KeyError: {e} - unique_id {unique_id} not found in file {file_path}")
        except OSError as e:
            print(f"Failed to open file {self.hdf5_dir}: {e}")

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    spectrum_ids, fluxes, sigmas, masks, wavelengths, latent_codes, metadatas = zip(*batch)
    
  
    # Pad sequences
    max_len = 26000
    
    def pad_trim(tensor):
        if len(tensor) > max_len:
            return tensor[:max_len]
        else:
            return torch.nn.functional.pad(tensor, (0, max_len - len(tensor)), value=0)
    
    fluxes_padded = torch.stack([pad_trim(f) for f in fluxes])
    sigmas_padded = torch.stack([pad_trim(s) for s in sigmas])
    masks_padded = torch.stack([pad_trim(m) for m in masks])
    wavelengths_padded = torch.stack([pad_trim(w) for w in wavelengths])
    
    # Create a new mask that considers both the original mask and padding
    new_masks = torch.zeros_like(fluxes_padded, dtype=torch.bool)
    for i, length in enumerate(map(len, fluxes)):
        new_masks[i, :min(length, max_len)] = masks[i][:min(length, max_len)]
    
    lengths = torch.tensor([min(len(f), max_len) for f in fluxes])
    
    return {
        'spectrum_id': spectrum_ids,
        'flux': fluxes_padded,
        'sigma': sigmas_padded,
        'mask': new_masks,
        'wavelength': wavelengths_padded,
        'latent_code': torch.stack(latent_codes),
        'metadata': metadatas,
        'lengths': lengths  # Original lengths before padding, capped at max_len
    }

