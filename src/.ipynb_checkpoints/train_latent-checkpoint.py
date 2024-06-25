import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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

# Custom modules
from dataset3 import APOGEEDataset
from model2 import Generator
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
    # Check if file exists to write headers accordingly
    file_exists = os.path.isfile(filepath)

    try:
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            # Write headers if file does not exist
            if not file_exists:
                writer.writerow(['Epoch', 'Total Training Time', 'Total Validation Time', 'Weight Optimization Time', 'Latent Optimization Time'])
            
            # Write data
            for data in timing_data:
                writer.writerow(data)
    
    except OSError as e:
        print(f"Failed to open file {filepath}: {e}")


def load_latent_vectors(hdf5_path, loaders, latent_dim, device):
    latent_vectors = []
    dict_latent_codes = {}  # Dictionary to store the mapping from unique_index to tensor index


    try:
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            idx = 0
            for loader in loaders:  # Loop over both train_loader and val_loader
                for batch in loader:
                    unique_ids = batch['unique_id'] 
                    for unique_id in unique_ids:
                        dict_latent_codes[unique_id] = idx
                        if unique_id in hdf5_file:
                            data = torch.tensor(hdf5_file[unique_id]['latent_code'][()], dtype=torch.float32, device=device)
                            latent_vectors.append(data)
                        else:
                            # Append zeros if no data found
                            latent_vectors.append(torch.zeros(latent_dim, device=device))
                        idx += 1
    except OSError as e:
        print(f"Failed to open file {hdf5_path}: {e}")
    # Stack all vectors to create the final tensor
    latent_codes = torch.stack(latent_vectors, dim=0).requires_grad_(True)
    return latent_codes, dict_latent_codes




def save_latent_vectors_to_hdf5(hdf5_path, dict_latent_codes, latent_vectors, epoch):

    try:
        with h5py.File(hdf5_path, 'a') as hdf5_file:
            for unique_id, index in dict_latent_codes.items():
                decoded_id = unique_id.decode('utf-8') if isinstance(unique_id, bytes) else unique_id
                versioned_key = f"{decoded_id}/optimized_latent_code/epoch_{epoch}"
                # versioned_key = f"{unique_id}/latent_code_epoch_{epoch}"
                # print(versioned_key)
                if versioned_key in hdf5_file:
                    del hdf5_file[versioned_key]  # Remove the old dataset if it exists
                hdf5_file[versioned_key] = latent_vectors[index].cpu().detach().numpy()  # Create a new dataset
                # print("saved  ", versioned_key)
    except OSError as e:
        print(f"Failed to open file {hdf5_path}: {e}")


def save_last_latent_vectors_to_hdf5(hdf5_path, dict_latent_codes, latent_vectors):

    try:
        with h5py.File(hdf5_path, 'a') as hdf5_file:
            for unique_id, index in dict_latent_codes.items():
                decoded_id = unique_id.decode('utf-8') if isinstance(unique_id, bytes) else unique_id
                
                versioned_key = f"{decoded_id}/optimized_latent_code/latest"
                if versioned_key in hdf5_file:
                    del hdf5_file[versioned_key]  # Remove the old dataset if it exists
                hdf5_file[versioned_key] = latent_vectors[index].cpu().detach().numpy()  # Create a new dataset
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

    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    latent_learning_rate = config['training']['latent_learning_rate']
    latent_dim = config['training']['latent_dim']
    checkpoint_interval = config['training']['checkpoint_interval']
    max_files=config['training']['max_files']

    return (config, data_path, checkpoints_path, latent_path, plots_path,
            batch_size, num_workers, num_epochs, learning_rate, latent_learning_rate, latent_dim, checkpoint_interval,dataset_name,max_files)

def prepare_datasets(config, data_path, max_files):
    print("Starting dataset...")
    dataset = APOGEEDataset(data_path, max_files)
    
    print(f"Loaded: {config['training']['max_files']} .. start split")
    train_indices, val_indices = train_test_split(list(range(max_files)), test_size=config['training']['split_ratios'][1])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)

    return train_loader, val_loader

def initialize_models_and_optimizers(config, train_loader,latent_codes, device):
    generator = Generator(config['training']['latent_dim'], config['model']['output_dim'], config['model']['generator_layers'], config['model']['activation_function']).to(device)
    # latent_codes = torch.randn(config['training']['max_files'], config['training']['latent_dim'], requires_grad=True, device=device)

    optimizer_g = optim.Adam(generator.parameters(), lr=config['training']['learning_rate'])
    optimizer_l = optim.Adam([latent_codes], lr=config['training']['latent_learning_rate'])

    return generator, optimizer_g, optimizer_l

def load_checkpoints(generator, latent_codes, optimizer_g, optimizer_l, checkpoints_path, train_loader, config, device):
    latest_checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_latest.pth.tar')
    best_checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_best.pth.tar')

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
        latent_codes = torch.randn(len(train_loader.dataset), config['training']['latent_dim'], device=device, requires_grad=True)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=config['training']['learning_rate'])
        optimizer_l = torch.optim.Adam([latent_codes], lr=config['training']['latent_learning_rate'])
        start_epoch = 0

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=config['training']['scheduler_step_size'], gamma=config['training']['scheduler_gamma'])
    scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_l, step_size=config['training']['scheduler_step_size'], gamma=config['training']['scheduler_gamma'])

    best_checkpoint = load_checkpoint(best_checkpoint_path)
    if best_checkpoint:
        try:
            best_val_loss = best_checkpoint['best_val_loss']
            print(f"Best validation loss from checkpoint: {best_val_loss}")
        except KeyError as e:
            print(f"Error retrieving best validation loss from checkpoint: {e}")
            best_val_loss = float('inf')
    else:
        best_val_loss = float('inf')

    return start_epoch, scheduler_g, scheduler_l, best_val_loss



def train_and_validate(generator, latent_codes, dict_latent_codes, optimizer_g, optimizer_l, scheduler_g, scheduler_l, train_loader, val_loader, start_epoch, num_epochs, checkpoints_path, latent_path, device, best_val_loss, data_path, plots_path):
    loss_history = {
        'train': [],
        'val': [],
        'before':[]
    }
    
    timing_data = []

    for epoch in range(start_epoch, num_epochs):

        
        epoch_train_time_start = time.time()
        epoch_train_time = 0
        epoch_val_time = 0
        epoch_weight_opt_time = 0
        epoch_latent_opt_time = 0

        
        generator.train()
        train_start_time = time.time()  # Time the entire training epoch
        epoch_losses = []
        epoch_losses_before = []
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")

        for batch in train_bar:
            batch_start_time = time.time()  # Time the processing of each batch

            unique_ids = batch['unique_id'] 
            flux = batch['flux'].to(device)
            mask = batch['flux_mask'].to(device)
            sigma = batch['sigma'].to(device)
            sigma_safe = sigma**2 + 1e-5
            indices = [dict_latent_codes[uid] for uid in unique_ids]

            # Optimize generator weights
            optimizer_g_start_time = time.time()
            latent_codes.requires_grad_(False)
            optimizer_g.zero_grad()
            generated = generator(latent_codes[indices])
            loss_g = weighted_mse_loss(generated, flux, mask/sigma_safe)
            loss_g.backward()
            optimizer_g.step()
            epoch_weight_opt_time += time.time() - optimizer_g_start_time
            epoch_losses_before.append(loss_g.item())
            

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

            epoch_losses.append(loss_l.item())
            

            # batch_time = time.time() - batch_start_time
            
            # train_bar.set_postfix({"Batch Total Time": batch_time, "Batch Loss": loss_l.item(), "Optimizer G Time": epoch_weight_opt_time, "Optimizer L Time": epoch_latent_opt_time})
        average_train_loss_before = np.mean(epoch_losses_before)
        average_train_loss = np.mean(epoch_losses)
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
            unique_ids = batch['unique_id']
            flux = batch['flux'].to(device)
            mask = batch['flux_mask'].to(device)
            sigma = batch['sigma'].to(device)
            sigma_safe = sigma**2 + 1e-5
            indices = [dict_latent_codes[uid] for uid in unique_ids]


            # Optimize latent codes
         
            for param in generator.parameters():
                param.requires_grad = False
            latent_codes.requires_grad_(True)
            optimizer_l.zero_grad()
            generated = generator(latent_codes[indices])
            val_loss = weighted_mse_loss(generated, flux, mask/sigma_safe)
            val_loss.backward()
            optimizer_l.step()
            for param in generator.parameters():
                param.requires_grad = True

            val_losses.append(val_loss.item())
        
            # # Ensure generator is in evaluation mode and doesn't track gradients
            # for param in generator.parameters():
            #     param.requires_grad = False
        
            # # Optimize latent codes
            # latent_codes.requires_grad_(True)
            # optimizer_l.zero_grad()
            # generated = generator(latent_codes[indices])
            # val_latent_loss = weighted_mse_loss(generated, flux, mask/sigma_safe)
            # val_latent_loss.backward()
            # optimizer_l.step()
            # latent_codes.requires_grad_(False)  # Turn off gradients for latent codes after update
        
            # val_losses.append(val_latent_loss.item())
            # val_bar.set_postfix({"Val Loss": val_latent_loss.item()})
            # average_val_loss = np.mean(val_losses)
            # loss_history['val'].append(average_val_loss)
            # print(f'Epoch {epoch+1} Average Validation Loss: {average_val_loss:.4f}')

        average_val_loss = np.mean(val_losses)
        epoch_val_time = time.time() - validation_start_time 
        # print(f'Epoch {epoch+1} Average Validation Loss: {average_val_loss:.4f}')

        # print(f'Epoch {epoch+1} Average Train Loss: {np.mean(epoch_losses):.4f}, Average Validation Loss: {average_val_loss:.4f}')
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

    save_last_latent_vectors_to_hdf5(data_path, dict_latent_codes, latent_codes)  # Save the final state of latent codes
    np.save(os.path.join(checkpoints_path, 'loss_history.npy'), loss_history)
    save_timing_data(os.path.join(plots_path, f'timing_data.csv'), timing_data)


def plot_losses(loss_history_path,plots_path):
    loss_history = np.load(loss_history_path, allow_pickle=True).item()
    train_losses = loss_history['train']
    val_losses = loss_history['val']
    before_losses= loss_history['before']

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(before_losses, label='Training Loss before latent optimization')
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0,10)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, f'lossesandbeforeopt.png'))

    # plt.show()



def main():
    print("Initializing device...")
    device = initialize_device()

    print("Loading configuration...")
    (config, data_path, checkpoints_path, latent_path, plots_path, 
     batch_size, num_workers, num_epochs, learning_rate, latent_learning_rate, latent_dim, checkpoint_interval, dataset_name, max_files) = load_configurations()

    print("Preparing datasets...")
    train_loader, val_loader = prepare_datasets(config, data_path, max_files)

    print("Compiling latent vectors...")
    loaders = [train_loader, val_loader]
    latent_vectors, dict_latent_codes = load_latent_vectors(data_path, loaders, latent_dim, device)

    print("Initializing models and optimizers...")
    generator, optimizer_g, optimizer_l = initialize_models_and_optimizers(config, train_loader, latent_vectors, device)

    print("Loading checkpoints...")
    start_epoch, scheduler_g, scheduler_l, best_val_loss = load_checkpoints(generator, latent_vectors, optimizer_g, optimizer_l, checkpoints_path, train_loader, config, device)

    print("Starting training and validation...")
    train_and_validate(generator, latent_vectors, dict_latent_codes, optimizer_g, optimizer_l, scheduler_g, scheduler_l, train_loader, val_loader, start_epoch, num_epochs, checkpoints_path, latent_path, device, best_val_loss, data_path, plots_path)

    print("Plotting losses...")
    plot_losses(os.path.join(checkpoints_path, 'loss_history.npy'), plots_path)
    

    print("Done.")

if __name__ == "__main__":
    main()

