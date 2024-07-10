import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from datetime import datetime
import numpy as np
from utils import get_config2, resolve_path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py
import csv
import time
import glob
# Custom modules
from multi_iter_dataset import IterableSpectraDataset, collate_fn
from model3 import Generator
from tqdm import tqdm
from checkpoint import save_checkpoint, load_checkpoint

def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def weighted_mse_loss(input, target, weight):
    assert input.shape == target.shape == weight.shape, f'Shapes of input {input.shape}, target {target.shape}, and weight {weight.shape} must match'
    loss = torch.mean(weight * (input - target) ** 2) 
    return loss

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

def load_configurations():
    config = get_config2()
    dataset_name = config['dataset_name']
    dataset_config = config['datasets'][dataset_name]
    data_path = resolve_path(dataset_config['path'])
    checkpoints_path = resolve_path(config['paths']['checkpoints'])
    latent_path = resolve_path(config['paths']['latent'])
    plots_path = resolve_path(config['paths']['plots'])
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    latent_learning_rate = config['training']['latent_learning_rate']
    latent_dim = config['training']['latent_dim']
    checkpoint_interval = config['training']['checkpoint_interval']
    max_files = config['training']['max_files']
    n_subspectra = config['training']['n_subspectra']
    return (config, data_path, checkpoints_path, latent_path, plots_path,
            batch_size, num_workers, num_epochs, learning_rate, latent_learning_rate, latent_dim, checkpoint_interval, dataset_name, max_files, n_subspectra)

def prepare_datasets(config, data_path):
    n_subspectra = config['training']['n_subspectra']
    train_dataset = IterableSpectraDataset(data_path, is_validation=False)
    val_dataset = IterableSpectraDataset(data_path, is_validation=True)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], collate_fn=collate_fn, num_workers=config['training']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], collate_fn=collate_fn, num_workers=config['training']['num_workers'], pin_memory=True)
    return train_loader, val_loader

def initialize_optimizers(config, generator, latent_codes):
    optimizer_g = optim.Adam(generator.parameters(), lr=config['training']['learning_rate'])
    optimizer_l = optim.Adam([latent_codes], lr=config['training']['latent_learning_rate'])
    return optimizer_g, optimizer_l

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoints(generator, optimizer_g, optimizer_l, checkpoints_path, config, device, train_loader):
    latest_checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_latest.pth.tar')
    best_checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_best.pth.tar')
    latest_checkpoint = load_checkpoint(latest_checkpoint_path)
    
    start_epoch = 0
    best_val_loss = float('inf')
    latent_codes = None
    dict_latent_codes = None

    if latest_checkpoint:
        try:
            generator.load_state_dict(latest_checkpoint['state_dict'])
            optimizer_g.load_state_dict(latest_checkpoint['optimizer_g_state'])
            optimizer_l.load_state_dict(latest_checkpoint['optimizer_l_state'])
            latent_codes = latest_checkpoint['latent_codes'].to(device)
            dict_latent_codes = latest_checkpoint['dict_latent_codes']
            start_epoch = latest_checkpoint['epoch']
            best_val_loss = latest_checkpoint['best_loss']
        
            print(f"Loaded latest checkpoint. Starting from epoch {start_epoch}")
        except KeyError as e:
            print(f"Error loading state dictionaries from latest checkpoint: {e}")
            print("Initializing from scratch.")
    
    if latent_codes is None or dict_latent_codes is None:
        print("Loading latent codes from dataset...")
        latent_codes, dict_latent_codes = train_loader.dataset.load_latent_vectors([train_loader], config['training']['latent_dim'], device)
        
        # Recreate optimizer_l with the loaded latent_codes
        optimizer_l = optim.Adam([latent_codes], lr=config['training']['latent_learning_rate'])

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=config['training']['scheduler_step_size'], gamma=config['training']['scheduler_gamma'])
    scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_l, step_size=config['training']['scheduler_step_size'], gamma=config['training']['scheduler_gamma'])

    # Load best validation loss if not found in latest checkpoint
    if best_val_loss == float('inf'):
        best_checkpoint = load_checkpoint(best_checkpoint_path)
        if best_checkpoint:
            try:
                best_val_loss = best_checkpoint['best_loss']
                print(f"Best validation loss from checkpoint: {best_val_loss}")
            except KeyError as e:
                print(f"Error retrieving best validation loss from checkpoint: {e}")

    return start_epoch, scheduler_g, scheduler_l, best_val_loss, latent_codes, dict_latent_codes, optimizer_l


def plot_losses(loss_history_path, plots_path):
    loss_history = np.load(loss_history_path, allow_pickle=True).item()
    train_losses = loss_history['train']
    val_losses = loss_history['val']
    before_losses = loss_history['before']
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(before_losses, label='Training Loss before latent optimization')
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, f'lossesandbeforeopt.png'))


def train_and_validate(generator, latent_codes, dict_latent_codes, optimizer_g, optimizer_l, scheduler_g, scheduler_l, train_loader, val_loader, start_epoch, num_epochs, checkpoints_path, latent_path, device, best_val_loss, data_path, plots_path, config):
    loss_history = {
        'train': [],
        'val': [],
        'before':[]
    }
    timing_data = []

    for epoch in range(start_epoch, num_epochs):
        # Load checkpoint at the beginning of each epoch
        _, scheduler_g, scheduler_l, best_val_loss, latent_codes, dict_latent_codes, optimizer_l = load_checkpoints(
            generator, optimizer_g, optimizer_l, checkpoints_path, config, device, train_loader
        )

        print("dict = " , len(dict_latent_codes))
        epoch_train_time_start = time.time()
        epoch_train_time = 0
        epoch_val_time = 0
        epoch_weight_opt_time = 0
        epoch_latent_opt_time = 0
        generator.train()
        train_start_time = time.time()
        epoch_losses = []
        epoch_losses_before = []
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")
        for batch in train_bar:
            batch_start_time = time.time()
            unique_ids = batch['spectrum_id']
            
            # Check for missing IDs and add them to dict_latent_codes and latent_codes
            missing_ids = [uid for uid in unique_ids if uid not in dict_latent_codes]
            if missing_ids:
                new_latent_codes = []
                for uid in missing_ids:
                    new_index = latent_codes.size(0)
                    dict_latent_codes[uid] = new_index
                    
                    # Get missing latent codes from the batch
                    new_latent_vector = batch['latent_code'][unique_ids.index(uid)].to(device)
                    new_latent_codes.append(new_latent_vector)
                
                # Concatenate new latent codes
                new_latent_codes = torch.stack(new_latent_codes)
                latent_codes = torch.cat([latent_codes, new_latent_codes], dim=0)
                
                # Create a new tensor with requires_grad=True
                latent_codes = nn.Parameter(latent_codes.detach().clone(), requires_grad=True)
                
                # Update optimizer_l with new latent_codes
                optimizer_l = optim.Adam([latent_codes], lr=optimizer_l.param_groups[0]['lr'])
            
            indices = torch.tensor([dict_latent_codes[uid] for uid in unique_ids], device=device)

            flux = batch['flux'].to(device)
            mask = batch['mask'].to(device)
            sigma = batch['sigma'].to(device)
            sigma_safe = sigma**2 + 1e-5
            
            optimizer_g_start_time = time.time()
            latent_codes.requires_grad_(False)
            optimizer_g.zero_grad()
            generated = generator(latent_codes[indices])
            loss_g = weighted_mse_loss(generated, flux, mask/sigma_safe)
            loss_g.backward()
            optimizer_g.step()
            epoch_weight_opt_time += time.time() - optimizer_g_start_time
            epoch_losses_before.append(loss_g.item())

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
            epoch_losses.append(loss_l.item())
        average_train_loss_before = np.mean(epoch_losses_before)
        average_train_loss = np.mean(epoch_losses)
        loss_history['train'].append(average_train_loss)
        loss_history['before'].append(average_train_loss_before)
        print(f'Epoch {epoch+1} Average Train Loss before latent: {average_train_loss_before:.4f}')
        print(f'Epoch {epoch+1} Average Train Loss after latent: {average_train_loss:.4f}')
        epoch_train_time = time.time() - epoch_train_time_start

        validation_start_time = time.time()
        generator.eval()
        val_losses = []
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")
        for batch in val_bar:
            unique_ids = batch['spectrum_id']
            flux = batch['flux'].to(device)
            mask = batch['mask'].to(device)
            sigma = batch['sigma'].to(device)
            sigma_safe = sigma**2 + 1e-5

            # Check for missing IDs and add them to dict_latent_codes and latent_codes
            missing_ids = [uid for uid in unique_ids if uid not in dict_latent_codes]
            for uid in missing_ids:
                new_index = len(latent_codes)
                dict_latent_codes[uid] = new_index
                
                # Get missing latent codes from the batch
                new_latent_vector = batch['latent_code'][unique_ids.index(uid)].to(device)
                latent_codes = torch.cat([latent_codes, new_latent_vector.unsqueeze(0)], dim=0)
                
            indices = [dict_latent_codes[uid] for uid in unique_ids]

            for param in generator.parameters():
                param.requires_grad = False
            latent_codes = latent_codes.detach().requires_grad_(True)
            optimizer_l.zero_grad()
            generated = generator(latent_codes[indices])
            val_loss = weighted_mse_loss(generated, flux, mask/sigma_safe)
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
            train_loader.dataset.save_latent_vectors_to_hdf5(dict_latent_codes, latent_codes, epoch + 1)
            print(f"Latent vectors saved for epoch {epoch + 1}")
        
        timing_data.append((epoch + 1, epoch_train_time, epoch_val_time, epoch_weight_opt_time, epoch_latent_opt_time))
        
        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
            'optimizer_g_state': optimizer_g.state_dict(),
            'optimizer_l_state': optimizer_l.state_dict(),
            'latent_codes': latent_codes.detach().cpu(),
            'dict_latent_codes': dict_latent_codes,
            'train_loss': average_train_loss,
            'val_loss': average_val_loss,
            'best_loss': best_val_loss
        }
        
        save_checkpoint(checkpoint_state, filename=os.path.join(checkpoints_path, 'checkpoint_latest.pth.tar'))
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            save_checkpoint(checkpoint_state, filename=os.path.join(checkpoints_path, 'checkpoint_best.pth.tar'))
        
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(checkpoint_state, filename=os.path.join(checkpoints_path, f'checkpoint_epoch_{epoch+1}.pth.tar'))

    torch.save({
        'latent_codes': latent_codes,
        'dict_latent_codes': dict_latent_codes
    }, os.path.join(latent_path, 'final_latent_codes.pth'))

    train_loader.dataset.save_last_latent_vectors_to_hdf5(dict_latent_codes, latent_codes)
    np.save(os.path.join(checkpoints_path, 'loss_history.npy'), loss_history)
    save_timing_data(os.path.join(plots_path, f'timing_data.csv'), timing_data)

    return latent_codes, dict_latent_codes  # Return the final versions

def main():
    print("Initializing device...")
    device = initialize_device()

    print("Loading configuration...")
    (config, data_path, checkpoints_path, latent_path, plots_path, 
     batch_size, num_workers, num_epochs, learning_rate, latent_learning_rate, latent_dim, checkpoint_interval, dataset_name, max_files, n_subspectra) = load_configurations()

    print("Preparing datasets...")
    train_loader, val_loader = prepare_datasets(config, data_path)

    print("Initializing model and optimizers...")
    generator = Generator(config['training']['latent_dim'], config['model']['output_dim'], config['model']['generator_layers'], config['model']['activation_function']).to(device)
    latent_codes = torch.randn(1, config['training']['latent_dim'], device=device)  # Initialize with a single random vector
    dict_latent_codes = {}
    optimizer_g = optim.Adam(generator.parameters(), lr=config['training']['learning_rate'])
    optimizer_l = optim.Adam([latent_codes], lr=config['training']['latent_learning_rate'])

    print("Loading checkpoints...")
    start_epoch, scheduler_g, scheduler_l, best_val_loss, latent_codes, dict_latent_codes, optimizer_l = load_checkpoints(
        generator, optimizer_g, optimizer_l, checkpoints_path, config, device, train_loader
    )

    print("Starting training and validation...")
    latent_codes, dict_latent_codes = train_and_validate(generator, latent_codes, dict_latent_codes, optimizer_g, optimizer_l, scheduler_g, scheduler_l, 
                       train_loader, val_loader, start_epoch, num_epochs, checkpoints_path, latent_path, device, 
                       best_val_loss, data_path, plots_path, config)
    print("Plotting losses...")
    plot_losses(os.path.join(checkpoints_path, 'loss_history.npy'), plots_path)
    
    print("Done.")

if __name__ == "__main__":
    main()
