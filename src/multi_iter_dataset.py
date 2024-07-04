import torch
from torch.utils.data import IterableDataset, DataLoader
import h5py
import numpy as np
import random
import os
from glob import glob

def ensure_native_byteorder(array):
    if array.dtype.byteorder not in ('=', '|'):
        return array.byteswap().newbyteorder()
    return array

class IterableSpectraDataset(IterableDataset):
    def __init__(self, hdf5_dir, n_samples_per_spectrum=500, validation_split=0.2, is_validation=False):
        self.hdf5_dir = hdf5_dir
        self.n_samples_per_spectrum = n_samples_per_spectrum
        self.validation_split = validation_split
        self.is_validation = is_validation
        self.file_list = glob(os.path.join(hdf5_dir, 'spectra_healpix_*.hdf5'))
        
        # Split files for cross-validation
        random.shuffle(self.file_list)
        split_idx = int(len(self.file_list) * (1 - validation_split))
        self.file_list = self.file_list[split_idx:] if is_validation else self.file_list[:split_idx]

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
            yield from self.process_file(file_path)

    def process_file(self, file_path):
        with h5py.File(file_path, 'r') as f:
            for group_name in f.keys():
                group = f[group_name]
                
                # Load data
                unique_id = group['unique_id'][()].decode('utf-8')
                flux = ensure_native_byteorder(group['flux'][:])
                wavelength = ensure_native_byteorder(group['wavelength'][:])
                sigma = ensure_native_byteorder(group['sigma'][:])
                mask = ensure_native_byteorder(group['flux_mask'][:])
                latent_code = ensure_native_byteorder(group['latent_code'][:])
                
                # Sample 500 random indices
                indices = np.random.choice(len(flux), self.n_samples_per_spectrum, replace=False)
                
                # Prepare tensors
                flux_tensor = torch.tensor(flux[indices], dtype=torch.float32)
                wavelength_tensor = torch.tensor(wavelength[indices], dtype=torch.float32)
                sigma_tensor = torch.tensor(sigma[indices], dtype=torch.float32)
                mask_tensor = torch.tensor(mask[indices], dtype=torch.float32)
                latent_code_tensor = torch.tensor(latent_code, dtype=torch.float32)
                
                # Prepare metadata
                metadata = {
                    'dec': group['dec'][()],
                    'instrument_type': group['instrument_type'][()].decode('utf-8'),
                    'instruments': group['instruments'][()],  # Keep as numpy array
                    'logg': group['logg'][()],
                    'logg_err': group['logg_err'][()],
                    'obj_class': group['obj_class'][()].decode('utf-8'),
                    'ra': group['ra'][()],
                    'rv': group['rv'][()],
                    'rv_err': group['rv_err'][()],
                    'temp': group['temp'][()],
                    'temp_err': group['temp_err'][()],
                }
                
                yield unique_id, flux_tensor, wavelength_tensor, sigma_tensor, mask_tensor, latent_code_tensor, metadata

def collate_fn(batch):
    unique_ids, fluxes, wavelengths, sigmas, masks, latent_codes, metadatas = zip(*batch)
    return {
        'unique_id': unique_ids,
        'flux': torch.stack(fluxes),
        'wavelength': torch.stack(wavelengths),
        'sigma': torch.stack(sigmas),
        'mask': torch.stack(masks),
        'latent_code': torch.stack(latent_codes),
        'metadata': metadatas
    }

def get_dataloaders(hdf5_dir, batch_size=32, n_samples_per_spectrum=500, validation_split=0.2, num_workers=4):
    train_dataset = IterableSpectraDataset(hdf5_dir, n_samples_per_spectrum, validation_split, is_validation=False)
    val_dataset = IterableSpectraDataset(hdf5_dir, n_samples_per_spectrum, validation_split, is_validation=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader

# # Example usage
# if __name__ == "__main__":
#     hdf5_dir = '../data/healpixfiles'
#     train_loader, val_loader = get_dataloaders(hdf5_dir)

#     # Example of iterating through the training data
#     for batch in train_loader:
#         unique_ids = batch['unique_id']
#         fluxes = batch['flux']
#         wavelengths = batch['wavelength']
#         sigmas = batch['sigma']
#         masks = batch['mask']
#         latent_codes = batch['latent_code']
#         metadatas = batch['metadata']

#         print(f"Batch size: {len(unique_ids)}")
#         print(f"Flux shape: {fluxes.shape}")
#         print(f"Wavelength shape: {wavelengths.shape}")
#         print(f"Sigma shape: {sigmas.shape}")
#         print(f"Mask shape: {masks.shape}")
#         print(f"Latent code shape: {latent_codes.shape}")
#         print(f"First spectrum ID: {unique_ids[0]}")
#         print(f"First spectrum temperature: {metadatas[0]['temp']}")

#         # Break after first batch for this example
#         break