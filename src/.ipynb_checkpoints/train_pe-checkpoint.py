import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
import numpy as np
from utils import get_config2, resolve_path
import matplotlib.pyplot as plt
import h5py
import csv
import time
from multi_iter_dataset import IterableSpectraDataset, collate_fn
from model_pe import Generator
from tqdm import tqdm
from checkpoint import save_checkpoint, load_checkpoint
import glob 

def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def weighted_mse_loss(input, target, weight):
    assert input.shape == target.shape == weight.shape, f'Shapes of input {input.shape}, target {target.shape}, and weight {weight.shape} must match'
    return torch.mean(weight * (input - target) ** 2)

def save_timing_data(filepath, timing_data):
    file_exists = os.path.isfile(filepath)
    try:
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Epoch', 'Total Training Time', 'Total Validation Time', 'Weight Optimization Time', 'Latent Optimization Time'])
            for data in timing_data:
                writer.writerow(data)
    except OSError as e:
        print(f"Failed to open file {filepath}: {e}")

def load_latent_vectors(data_dir, latent_dim, device):
    latent_vectors = []
    dict_latent_codes = {}
    idx = 0

    # Get all HDF5 files in the directory
    hdf5_files = glob.glob(os.path.join(data_dir, '*.hdf5'))

    for hdf5_path in hdf5_files:
        try:
            with h5py.File(hdf5_path, 'r') as hdf5_file:
                for spectrum_id in hdf5_file.keys():
                    if 'LATENT' in hdf5_file[spectrum_id]:
                        data = torch.tensor(hdf5_file[spectrum_id]['LATENT'][()], dtype=torch.float32, device=device)
                        latent_vectors.append(data)
                    else:
                        latent_vectors.append(torch.randn(latent_dim, device=device))
                    
                    dict_latent_codes[spectrum_id] = idx
                    idx += 1
        except OSError as e:
            print(f"Failed to open file {hdf5_path}: {e}")

    if not latent_vectors:
        print("No latent vectors found. Initializing with random vectors.")
        return torch.randn(idx, latent_dim, device=device), dict_latent_codes

    latent_codes = torch.stack(latent_vectors, dim=0).requires_grad_(True)
    return latent_codes, dict_latent_codes

def save_latent_vectors_to_hdf5(data_dir, dict_latent_codes, latent_vectors, epoch):
    hdf5_files = glob.glob(os.path.join(data_dir, '*.hdf5'))
    
    for hdf5_path in hdf5_files:
        try:
            with h5py.File(hdf5_path, 'a') as hdf5_file:
                for unique_id, index in dict_latent_codes.items():
                    if unique_id in hdf5_file:
                        versioned_key = f"{unique_id}/optimized_latent_code/epoch_{epoch}"
                        if versioned_key in hdf5_file:
                            del hdf5_file[versioned_key]
                        hdf5_file[versioned_key] = latent_vectors[index].cpu().detach().numpy()
        except OSError as e:
            print(f"Failed to open file {hdf5_path}: {e}")

def save_last_latent_vectors_to_hdf5(data_dir, dict_latent_codes, latent_vectors):
    hdf5_files = glob.glob(os.path.join(data_dir, '*.hdf5'))
    
    for hdf5_path in hdf5_files:
        try:
            with h5py.File(hdf5_path, 'a') as hdf5_file:
                for unique_id, index in dict_latent_codes.items():
                    if unique_id in hdf5_file:
                        versioned_key = f"{unique_id}/optimized_latent_code/latest"
                        if versioned_key in hdf5_file:
                            del hdf5_file[versioned_key]
                        hdf5_file[versioned_key] = latent_vectors[index].cpu().detach().numpy()
        except OSError as e:
            print(f"Failed to open file {hdf5_path}: {e}")

def load_configurations():
    config = get_config2()
    dataset_name = config['dataset_name']
    dataset_config = config['datasets'][dataset_name]
    data_path = resolve_path(dataset_config['path'])
    checkpoints_path = resolve_path(config['paths']['checkpoints'])
    latent_path = resolve_path(config['paths']['latent'])
    plots_path = resolve_path(config['paths']['plots'])

    return config, data_path, checkpoints_path, latent_path, plots_path

def prepare_datasets(config, data_path):
    train_dataset = IterableSpectraDataset(data_path, n_samples_per_spectrum=config['training']['n_samples_per_spectrum'], is_validation=False)
    val_dataset = IterableSpectraDataset(data_path, n_samples_per_spectrum=config['training']['n_samples_per_spectrum'], is_validation=True)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'], collate_fn=collate_fn)

    return train_loader, val_loader

def initialize_models_and_optimizers(config, latent_codes, device):
    generator = Generator(config['training']['latent_dim'], config['model']['output_dim'], 
                          config['model']['generator_layers'], config['model']['activation_function']).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=config['training']['learning_rate'])
    optimizer_l = optim.Adam([latent_codes], lr=config['training']['latent_learning_rate'])

    return generator, optimizer_g, optimizer_l

def load_checkpoints(generator, latent_codes, optimizer_g, optimizer_l, checkpoints_path, config, device):
    latest_checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_latest.pth.tar')
    best_checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_best.pth.tar')

    start_epoch = 0
    best_val_loss = float('inf')

    latest_checkpoint = load_checkpoint(latest_checkpoint_path)
    if latest_checkpoint:
        try:
            generator.load_state_dict(latest_checkpoint['generator_state'])
            latent_codes.data = latest_checkpoint['latent_codes']
            optimizer_g.load_state_dict(latest_checkpoint['optimizer_g_state'])
            optimizer_l.load_state_dict(latest_checkpoint['optimizer_l_state'])
            start_epoch = latest_checkpoint['epoch'] + 1
            print("Loaded latest checkpoint.")
        except KeyError as e:
            print(f"Error loading state dictionaries from latest checkpoint: {e}")
            start_epoch = 0
    else:
        generator.apply(Generator.init_weights)
        print("No checkpoint found. Starting from scratch.")

    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=config['training']['scheduler_step_size'], gamma=config['training']['scheduler_gamma'])
    scheduler_l = optim.lr_scheduler.StepLR(optimizer_l, step_size=config['training']['scheduler_step_size'], gamma=config['training']['scheduler_gamma'])

    best_checkpoint = load_checkpoint(best_checkpoint_path)
    if best_checkpoint:
        try:
            best_val_loss = best_checkpoint['best_val_loss']
            print(f"Best validation loss from checkpoint: {best_val_loss}")
        except KeyError as e:
            print(f"Error retrieving best validation loss from checkpoint: {e}")
            best_val_loss = float('inf')

    return start_epoch, scheduler_g, scheduler_l, best_val_loss

def train_and_validate(generator, latent_codes, dict_latent_codes, optimizer_g, optimizer_l, scheduler_g, scheduler_l, 
                       train_loader, val_loader, start_epoch, num_epochs, checkpoints_path, latent_path, device, 
                       best_val_loss, data_path, plots_path):
    loss_history = {'train': [], 'val': [], 'before': []}
    timing_data = []

    for epoch in range(start_epoch, num_epochs):
        epoch_train_time_start = time.time()
        epoch_train_time = epoch_val_time = epoch_weight_opt_time = epoch_latent_opt_time = 0

        # Training phase
        generator.train()
        train_losses = []
        train_losses_before = []
        print(f"Starting training epoch {epoch + 1}")  # Added for debugging
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")

        first_batch_printed = False  # Added for debugging

        for batch in train_bar:
            if not first_batch_printed:  # Added for debugging
                print("First batch contents:")  # Added for debugging
                print(f"spectrum_ids: {batch['spectrum_id']}")  # Added for debugging
                print(f"flux: {batch['flux']}")  # Added for debugging
                print(f"wavelength: {batch['wavelength']}")  # Added for debugging
                print(f"sigma: {batch['sigma']}")  # Added for debugging
                print(f"mask: {batch['mask']}")  # Added for debugging
                first_batch_printed = True  # Added for debugging
                break  # Added for debugging

            spectrum_ids = batch['spectrum_id']
            flux = batch['flux'].to(device)
            wavelength = batch['wavelength'].to(device)
            sigma = batch['sigma'].to(device)
            mask = batch['mask'].to(device)
            sigma_safe = sigma**2 + 1e-5

            indices = [dict_latent_codes[sid] for sid in spectrum_ids]

            # Optimize generator weights
            optimizer_g_start_time = time.time()
            latent_codes.requires_grad_(False)
            optimizer_g.zero_grad()
            generated = generator(latent_codes[indices])
            loss_g = weighted_mse_loss(generated, flux, mask/sigma_safe)
            loss_g.backward()
            optimizer_g.step()
            epoch_weight_opt_time += time.time() - optimizer_g_start_time
            train_losses_before.append(loss_g.item())

            # Optimize latent codes
            optimizer_l_start_time = time.time()
            for param in generator.parameters():
                param.requires_grad = False
            latent_codes.requires_grad_(True)
            optimizer_l.zero_grad()
            generated = generator(latent_codes[indices])
            loss_l = weighted_mse_loss(generated, flux, mask/sigma_safe)
            loss_l.backward()
            optimizer_l.step()
            for param in generator.parameters():
                param.requires_grad = True
            epoch_latent_opt_time += time.time() - optimizer_l_start_time

            train_losses.append(loss_l.item())

        if not first_batch_printed:  # Added for debugging
            print("No batches were loaded in this epoch.")  # Added for debugging
            return  # Exit the function early for debugging

        average_train_loss_before = np.mean(train_losses_before)
        average_train_loss = np.mean(train_losses)
        loss_history['train'].append(average_train_loss)
        loss_history['before'].append(average_train_loss_before)
            
        print(f'Epoch {epoch+1} Average Train Loss before latent: {average_train_loss_before:.4f}')
        print(f'Epoch {epoch+1} Average Train Loss after latent: {average_train_loss:.4f}')

        epoch_train_time = time.time() - epoch_train_time_start

        # Validation phase
        validation_start_time = time.time()
        generator.eval()
        val_losses = []
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")

        for batch in val_bar:
            spectrum_ids = batch['spectrum_id']
            flux = batch['flux'].to(device)
            wavelength = batch['wavelength'].to(device)
            sigma = batch['sigma'].to(device)
            mask = batch['mask'].to(device)
            sigma_safe = sigma**2 + 1e-5

            # Debugging statements
            print(f"Validation Epoch {epoch+1}, Batch: {spectrum_ids}")
            print(f"Validation Flux shape: {flux.shape}, Wavelength shape: {wavelength.shape}, Sigma shape: {sigma.shape}, Mask shape: {mask.shape}")

            indices = [dict_latent_codes[sid] for sid in spectrum_ids]

            # Optimize latent codes
            for param in generator.parameters():
                param.requires_grad = False
            latent_codes.requires_grad_(True)
            optimizer_l.zero_grad()
            generated = generator(latent_codes[indices])
            val_loss = weighted_mse_loss(generated, flux, mask/sigma_safe)
            print(f"Validation Loss: {val_loss.item()}")  # Debugging statement
            val_loss.backward()
            optimizer_l.step()
            for param in generator.parameters():
                param.requires_grad = True

            val_losses.append(val_loss.item())

        average_val_loss = np.mean(val_losses)
        epoch_val_time = time.time() - validation_start_time 
        print(f'Epoch {epoch+1} Average Validation Loss: {average_val_loss:.4f}')
        loss_history['val'].append(average_val_loss)

        scheduler_g.step()
        scheduler_l.step()

        if (epoch + 1) % 5 == 0 or epoch + 1 == num_epochs:
            save_latent_vectors_to_hdf5(data_path, dict_latent_codes, latent_codes, epoch + 1)
            print(f"Latent vectors saved for epoch {epoch + 1}")
        
        timing_data.append((epoch + 1, epoch_train_time, epoch_val_time, epoch_weight_opt_time, epoch_latent_opt_time))

        # Save checkpoints
        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
            'optimizer_g_state': optimizer_g.state_dict(),
            'optimizer_l_state': optimizer_l.state_dict(),
            'latent_codes': latent_codes.detach().cpu(),
            'train_loss': average_train_loss,
            'val_loss': average_val_loss,
            'best_loss': best_val_loss
        }
        save_checkpoint(checkpoint_state, filename=os.path.join(checkpoints_path, 'checkpoint_latest.pth.tar'))
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            save_checkpoint(checkpoint_state, filename=os.path.join(checkpoints_path, 'checkpoint_best.pth.tar'))
        save_checkpoint(checkpoint_state, filename=os.path.join(checkpoints_path, f'checkpoint_epoch_{epoch+1}.pth.tar'))

    save_last_latent_vectors_to_hdf5(data_path, dict_latent_codes, latent_codes)
    np.save(os.path.join(checkpoints_path, 'loss_history.npy'), loss_history)
    save_timing_data(os.path.join(plots_path, f'timing_data.csv'), timing_data)

    return loss_history, best_val_loss




def plot_losses(loss_history, plots_path):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history['train'], label='Training Loss')
    plt.plot(loss_history['val'], label='Validation Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, f'losses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    plt.close()

def main():
    print("Initializing...")
    device = initialize_device()

    print("Loading configuration...")
    config, data_path, checkpoints_path, latent_path, plots_path = load_configurations()

    print("Preparing datasets...")
    train_loader, val_loader = prepare_datasets(config, data_path)

    print("Loading latent vectors...")
    latent_codes, dict_latent_codes = load_latent_vectors(data_path, config['training']['latent_dim'], device)

    print("Initializing model...")
    generator, optimizer_g, optimizer_l = initialize_models_and_optimizers(config, latent_codes, device)

    print("Loading checkpoints...")
    start_epoch, scheduler_g, scheduler_l, best_val_loss = load_checkpoints(
        generator, latent_codes, optimizer_g, optimizer_l, checkpoints_path, config, device
    )

    print("Starting training...")
    loss_history, best_val_loss = train_and_validate(
        generator, latent_codes, dict_latent_codes, 
        optimizer_g, optimizer_l, scheduler_g, scheduler_l,
        train_loader, val_loader, start_epoch, 
        config['training']['num_epochs'], checkpoints_path, 
        latent_path, device, best_val_loss, data_path, plots_path
    )

    print("Training completed. Plotting losses...")
    plot_losses(loss_history, plots_path)

    print("Done.")

if __name__ == "__main__":
    main()
