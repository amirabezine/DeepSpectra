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
from dataset import IterableSpectraDataset, collate_fn
from model import Generator
from tqdm import tqdm



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

    return (config, data_path)


def prepare_datasets(config, data_path):
    n_samples_per_spectrum =  config['training']['n_samples_per_spectrum']
    n_subspectra = config['training']['n_subspectra']
    train_dataset = IterableSpectraDataset(data_path, is_validation=False, n_subspectra=n_subspectra)
    val_dataset = IterableSpectraDataset(data_path, is_validation=True, n_subspectra=n_subspectra)
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




