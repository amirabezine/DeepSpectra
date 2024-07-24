from torch.utils.data import DataLoader
from dataset import IterableSpectraDataset, collate_fn

class DataPreparation:
    def __init__(self, config):
        self.config = config

    def prepare_datasets(self):
        n_subspectra = self.config.n_subspectra
        train_dataset = IterableSpectraDataset(self.config.data_path, is_validation=False, n_subspectra=n_subspectra)
        val_dataset = IterableSpectraDataset(self.config.data_path, is_validation=True, n_subspectra=n_subspectra)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, collate_fn=collate_fn, num_workers=self.config.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, collate_fn=collate_fn, num_workers=self.config.num_workers, pin_memory=True)
        return train_loader, val_loader
