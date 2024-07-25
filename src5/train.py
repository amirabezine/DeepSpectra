import time
import numpy as np
import torch
from tqdm import tqdm
from loss import WeightedMSELoss
from checkpoint import CheckpointManager
from utils import save_timing_data
import os

class Trainer:
    def __init__(self, config, spectrum_generator, latent_codes, dict_latent_codes, optimizer_g, optimizer_l, scheduler_g, scheduler_l, train_loader, val_loader, start_epoch, device, best_val_loss):
        self.config = config
        self.spectrum_generator = spectrum_generator  # Using SpectrumGenerator instead of just Generator
        self.latent_codes = latent_codes
        self.dict_latent_codes = dict_latent_codes
        self.optimizer_g = optimizer_g
        self.optimizer_l = optimizer_l
        self.scheduler_g = scheduler_g
        self.scheduler_l = scheduler_l
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.start_epoch = start_epoch
        self.device = device
        self.best_val_loss = best_val_loss
        self.loss_fn = WeightedMSELoss()

    
    def handle_missing_ids(self, batch):
        unique_ids = batch['spectrum_id']
        missing_ids = [uid for uid in unique_ids if uid not in self.dict_latent_codes]
        if missing_ids:
            new_latent_codes = []
            for uid in missing_ids:
                new_index = self.latent_codes.size(0)
                self.dict_latent_codes[uid] = new_index

                # Get missing latent codes from the batch
                new_latent_vector = batch['latent_code'][unique_ids.index(uid)].to(self.device)
                new_latent_codes.append(new_latent_vector)

            # Concatenate new latent codes
            new_latent_codes = torch.stack(new_latent_codes)
            self.latent_codes = torch.cat([self.latent_codes, new_latent_codes], dim=0)

            # Create a new tensor with requires_grad=True
            self.latent_codes = torch.nn.Parameter(self.latent_codes.detach().clone(), requires_grad=True)

            # Update optimizer_l with new latent_codes
            self.optimizer_l = torch.optim.Adam([self.latent_codes], lr=self.optimizer_l.param_groups[0]['lr'])

    def compute_loss(self, generated, flux, mask, sigma_safe):
        return self.loss_fn.compute(generated, flux, mask / sigma_safe)

    
    def train_one_epoch(self, epoch, timing_data):
        self.spectrum_generator.train()
        epoch_losses = []
        epoch_losses_before = []
        epoch_train_time_start = time.time()
        epoch_weight_opt_time = 0
        epoch_latent_opt_time = 0
    
        train_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}/{self.config.num_epochs}")
        for batch in train_bar:
            self.handle_missing_ids(batch)
            observed_wavelengths = batch['wavelength'].to(self.device)
            indices = torch.tensor([self.dict_latent_codes[uid] for uid in batch['spectrum_id']], device=self.device)
            flux = batch['flux'].to(self.device)
            mask = batch['mask'].to(self.device)
            sigma = batch['sigma'].to(self.device)
            sigma_safe = sigma**2 + 1e-5
    
            # Generator step
            self.latent_codes.requires_grad_(False)
            for param in self.spectrum_generator.generator.parameters():
                param.requires_grad = True
            self.optimizer_g.zero_grad()
            generated_before = self.spectrum_generator(self.latent_codes[indices], observed_wavelengths)
            loss_g = self.compute_loss(generated_before, flux, mask, sigma_safe)
            loss_g.backward()
            self.optimizer_g.step()
            epoch_weight_opt_time += time.time() - time.time()
            epoch_losses_before.append(loss_g.item())
    
            # Latent codes step
            for param in self.spectrum_generator.generator.parameters():
                param.requires_grad = False
            self.latent_codes.requires_grad_(True)
            self.optimizer_l.zero_grad()
            generated_after = self.spectrum_generator(self.latent_codes[indices], observed_wavelengths)
            loss_l = self.compute_loss(generated_after, flux, mask, sigma_safe)
            loss_l.backward()
            print("Gradients on latent codes:", self.latent_codes.grad)
            print("Latent codes in optimizer:", list(self.optimizer_l.param_groups[0]['params'])[0].grad)

            # print("generated_after grad_fn:", generated_after.grad_fn)

            # print("Gradients on latent codes:", self.latent_codes.grad)  # Debugging print

            if generated_after.grad is not None:
                print("Gradients computed on generated output.")
            else:
                print("No gradients on generated output.")

            self.optimizer_l.step()
            for param in self.spectrum_generator.generator.parameters():
                param.requires_grad = True
            epoch_latent_opt_time += time.time() - time.time()
            epoch_losses.append(loss_l.item())
    
        average_train_loss_before = np.mean(epoch_losses_before)
        average_train_loss = np.mean(epoch_losses)
        timing_data.append((epoch + 1, time.time() - epoch_train_time_start, epoch_weight_opt_time, epoch_latent_opt_time))
        return average_train_loss, average_train_loss_before

    def validate_one_epoch(self, epoch):
        self.spectrum_generator.eval()
        val_losses = []
        validation_start_time = time.time()

        val_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch + 1}/{self.config.num_epochs}")
        for batch in val_bar:
            self.handle_missing_ids(batch)
            observed_wavelengths = batch['wavelength'].to(self.device)  # Handling observed wavelengths

            indices = [self.dict_latent_codes[uid] for uid in batch['spectrum_id']]
            flux = batch['flux'].to(self.device)
            mask = batch['mask'].to(self.device)
            sigma = batch['sigma'].to(self.device)
            sigma_safe = sigma**2 + 1e-5
            
            for param in self.spectrum_generator.generator.parameters():
                param.requires_grad = False  # Freeze generator weights
            self.latent_codes.requires_grad_(True)  # Activate latent codes

            self.optimizer_l.zero_grad()
            generated = self.spectrum_generator(self.latent_codes[indices], observed_wavelengths)  # Using new SpectrumGenerator
            val_loss = self.compute_loss(generated, flux, mask, sigma_safe)
            val_loss.backward()
            self.optimizer_l.step()
            val_losses.append(val_loss.item())

        average_val_loss = np.mean(val_losses)
        return average_val_loss, time.time() - validation_start_time


    def train_and_validate(self):
        loss_history = {
            'train': [],
            'val': [],
            'before': []
        }
        timing_data = []

        for epoch in range(self.start_epoch, self.config.num_epochs):
            checkpoint_manager = CheckpointManager(self.config, self.spectrum_generator, self.optimizer_g, self.optimizer_l, self.device, self.train_loader)
            _, self.scheduler_g, self.scheduler_l, self.best_val_loss, self.latent_codes, self.dict_latent_codes, self.optimizer_l = checkpoint_manager.load_checkpoints()

            average_train_loss, average_train_loss_before = self.train_one_epoch(epoch, timing_data)
            average_val_loss, epoch_val_time = self.validate_one_epoch(epoch)

            loss_history['train'].append(average_train_loss)
            loss_history['before'].append(average_train_loss_before)
            loss_history['val'].append(average_val_loss)

            print(f'Epoch {epoch + 1} Average Train Loss before latent: {average_train_loss_before:.4f}')
            print(f'Epoch {epoch + 1} Average Train Loss after latent: {average_train_loss:.4f}')
            print(f'Epoch {epoch + 1} Average Validation Loss: {average_val_loss:.4f}')

            self.scheduler_g.step()
            self.scheduler_l.step()

            if (epoch + 1) % 5 == 0 or epoch + 1 == self.config.num_epochs:
                self.train_loader.dataset.save_latent_vectors_to_hdf5(self.dict_latent_codes, self.latent_codes, epoch + 1)
                print(f"Latent vectors saved for epoch {epoch + 1}")

            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': self.spectrum_generator.state_dict(),
                'optimizer_g_state': self.optimizer_g.state_dict(),
                'optimizer_l_state': self.optimizer_l.state_dict(),
                'latent_codes': self.latent_codes.detach().cpu(),
                'dict_latent_codes': self.dict_latent_codes,
                'train_loss': average_train_loss,
                'val_loss': average_val_loss,
                'best_loss': self.best_val_loss
            }

            checkpoint_manager.save_checkpoint(checkpoint_state, filename=os.path.join(self.config.checkpoints_path, 'checkpoint_latest.pth.tar'))
            if average_val_loss < self.best_val_loss:
                self.best_val_loss = average_val_loss
                checkpoint_manager.save_checkpoint(checkpoint_state, filename=os.path.join(self.config.checkpoints_path, 'checkpoint_best.pth.tar'))

            if (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_manager.save_checkpoint(checkpoint_state, filename=os.path.join(self.config.checkpoints_path, f'checkpoint_epoch_{epoch + 1}.pth.tar'))

        torch.save({
            'latent_codes': self.latent_codes,
            'dict_latent_codes': self.dict_latent_codes
        }, os.path.join(self.config.latent_path, 'final_latent_codes.pth'))

        self.train_loader.dataset.save_last_latent_vectors_to_hdf5(self.dict_latent_codes, self.latent_codes)
        np.save(os.path.join(self.config.checkpoints_path, 'loss_history.npy'), loss_history)
        save_timing_data(os.path.join(self.config.plots_path, f'timing_data.csv'), timing_data)

        return self.latent_codes, self.dict_latent_codes  # Return the final versions
