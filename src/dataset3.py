import torch
from torch.utils.data import Dataset
import h5py

class APOGEEDataset(Dataset):
    def __init__(self, hdf5_file, max_files=None):
        self.hdf5_file = hdf5_file
        self.f = h5py.File(hdf5_file, 'r')
        self.files = list(self.f.keys())
        if max_files:
            self.files = self.files[:max_files]
        self.indices = self.get_indices()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        
        flux = self.f[file]['flux'][:]
        wavelength = self.f[file]['wavelength'][:]
        snr = self.f[file]['snr'][()]
        flux_mask = self.f[file]['flux_mask'][:]
        sigma = self.f[file]['sigma'][:]
        unique_id = self.f[file]['unique_id'][()]  # Read unique ID
        latent_code = self.f[file]['latent_code'][:]  # Read latent code
        index = self.f[file]['index'][()] 
        
        return {
            'unique_id': unique_id,
            'index': torch.tensor(index, dtype=torch.long),
            'flux': torch.tensor(flux, dtype=torch.float32),
            'wavelength': torch.tensor(wavelength, dtype=torch.float32),
            'snr': torch.tensor(snr, dtype=torch.float32),
            'flux_mask': torch.tensor(flux_mask, dtype=torch.float32),
            'sigma': torch.tensor(sigma, dtype=torch.float32),
            'latent_code': torch.tensor(latent_code, dtype=torch.float32)  # Return latent code
        }

    def __del__(self):
        self.f.close()

    def get_indices(self):
        indices = []
        for file in self.files:
            indices.append(self.f[file]['index'][()])
        return indices



def get_dataloaders(hdf5_file, batch_size, num_workers, split_ratios):
    dataset = APOGEEDataset(hdf5_file)
    lengths = [int(len(dataset) * ratio) for ratio in split_ratios]
    lengths[-1] = len(dataset) - sum(lengths[:-1]) 
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
