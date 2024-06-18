import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import os
from datetime import datetime
import numpy as np
from utils import get_config, resolve_path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Custom modules
from dataset2 import APOGEEDataset
from model2 import Generator
from tqdm import tqdm
from checkpoint import save_checkpoint, load_checkpoint


def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def weighted_mse_loss(input, target, weight):
    assert input.shape == target.shape == weight.shape, f'Shapes of input {input.shape}, target {target.shape}, and weight {weight.shape} must match'
    loss = torch.mean(weight * (input - target) ** 2) ###uncomment later, debugging
    # loss = torch.mean((input - target) ** 2)
    return loss

def load_configurations():
    config = get_config()
    data_path = resolve_path(config['paths']['hdf5_data'])
    checkpoints_path = resolve_path(config['paths']['checkpoints'])
    latent_path = resolve_path(config['paths']['latent'])
    plots_path = resolve_path(config['paths']['plots'])
    tensorboard_path = resolve_path(config['paths']['tensorboard'])

    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    latent_learning_rate = config['training']['latent_learning_rate']
    latent_dim = config['training']['latent_dim']
    checkpoint_interval = config['training']['checkpoint_interval']

    return (config, data_path, checkpoints_path, latent_path, plots_path, tensorboard_path,
            batch_size, num_workers, num_epochs, learning_rate, latent_learning_rate, latent_dim, checkpoint_interval)

def prepare_datasets(config, data_path):
    print("Starting dataset...")
    dataset = APOGEEDataset(data_path, max_files=config['training']['max_files'])
    print(f"Loaded: {config['training']['max_files']} .. start split")
    train_indices, val_indices = train_test_split(list(range(config['training']['max_files'])), test_size=config['training']['split_ratios'][1])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], pin_memory=True)

    return train_loader, val_loader

def initialize_models_and_optimizers(config, train_loader, device):
    generator = Generator(config['training']['latent_dim'], config['model']['output_dim'], config['model']['generator_layers'], config['model']['activation_function']).to(device)
    latent_codes = torch.randn(config['training']['max_files'], config['training']['latent_dim'], requires_grad=True, device=device)

    optimizer_g = optim.Adam(generator.parameters(), lr=config['training']['learning_rate'])
    optimizer_l = optim.Adam([latent_codes], lr=config['training']['latent_learning_rate'])

    return generator, latent_codes, optimizer_g, optimizer_l

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

def train_and_validate(generator, latent_codes, optimizer_g, optimizer_l, scheduler_g, scheduler_l, train_loader, val_loader, start_epoch, num_epochs, checkpoints_path, latent_path, device, best_val_loss):
    loss_history = {
        'train': [],
        'val': [],
        'before':[]
    }

    for epoch in range(start_epoch, num_epochs):
        generator.train()
        epoch_losses = []
        epoch_losses_before =[]
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")

        for batch in train_bar:
            indices = batch['index'].to(device, dtype=torch.long)
            flux = batch['flux'].to(device)
            mask = batch['flux_mask'].to(device)
            sigma = batch['sigma'].to(device)
            sigma_safe= sigma**2 + 1e-3
            
            # # Debug: Check latent codes require_grad status
            # print(f"Latent codes require_grad before optimization should be true 1: {latent_codes.requires_grad}")


            # Step 1: Optimize generator weights
            latent_codes.requires_grad_(False)  # Freeze latent codes  ####### error probably here
             # Debug: Check latent codes require_grad status
            # print(f"Latent codes require_grad before optimization should be false 1: {latent_codes.requires_grad}")
            optimizer_g.zero_grad()
            generated = generator(latent_codes[indices])
            # print(f"Generated sample outputs: {generated[:5]}")  # Print first few outputs
            # print(mask/sigma_safe)
            loss_g = weighted_mse_loss(generated, flux, mask/sigma_safe)
            loss_g.backward()
            optimizer_g.step()  
            epoch_losses_before.append(loss_g.item())
            train_bar.set_postfix({"Batch Weight Loss": loss_g.item()})
            # # Debug: Print gradients of latent codes after optimizing generator
            # if latent_codes.grad is not None:
            #     print(f"Gradients of latent codes after generator update: {latent_codes.grad}")


            # Step 2: Freeze generator weights and optimize latent codes
            for param in generator.parameters():
                param.requires_grad = False
            # # Debug: Check if latent codes updated
            # print(f"Latent codes require_grad after optimization should be false 2: {latent_codes.requires_grad}")
            latent_codes.requires_grad_(True)  # Unfreeze latent codes   
            optimizer_l.zero_grad()
            generated = generator(latent_codes[indices])
            loss_l = weighted_mse_loss(generated, flux, mask/sigma_safe)
            
            loss_l.backward()
            # Debug: Print gradients of latent codes before optimizer step
            # print(f"Gradients of latent codes before optimizer_l step: {latent_codes.grad}")

            
            optimizer_l.step()
            # Debug: Check if latent codes updated
            # print(f"Latent codes require_grad after optimization should be true2: {latent_codes.requires_grad}")

            
            for param in generator.parameters():
                param.requires_grad = True  # Unfreeze generator weights

            epoch_losses.append(loss_l.item())
            train_bar.set_postfix({"Batch Latent Loss": loss_l.item()})
            
            # # Debug: Print latent codes to check updates
            # if epoch == 2:  # Example: Print only for the first epoch to reduce output
            #     print(f"Latent codes sample values after update: {latent_codes.data[:5]}")

            
        average_train_loss_before = np.mean(epoch_losses_before)
        average_train_loss = np.mean(epoch_losses)
        loss_history['train'].append(average_train_loss)
        loss_history['before'].append(average_train_loss_before)
        print(f'Epoch {epoch+1} Average Train Loss before latent: {average_train_loss_before:.4f}')
        print(f'Epoch {epoch+1} Average Train Loss after latent: {average_train_loss:.4f}')

        generator.eval()
        val_losses = []
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")
        with torch.no_grad():
            for batch in val_bar:
                indices = batch['index'].to(device, dtype=torch.long)
                flux = batch['flux'].to(device)
                mask = batch['flux_mask'].to(device)
                sigma = batch['sigma'].to(device)
                sigma_safe= sigma**2 + 1e-3

                generated = generator(latent_codes[indices])
                val_loss = weighted_mse_loss(generated, flux, mask/sigma_safe)
                val_losses.append(val_loss.item())
                val_bar.set_postfix({"Batch Val Loss": val_loss.item()})

        average_val_loss = np.mean(val_losses)
        loss_history['val'].append(average_val_loss)
        print(f'Epoch {epoch+1} Average Validation Loss: {average_val_loss:.4f}')

        # Step the schedulers
        scheduler_g.step()  
        scheduler_l.step()  
        

        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
            'latent_codes': latent_codes,
            'optimizer_g_state': optimizer_g.state_dict(),
            'optimizer_l_state': optimizer_l.state_dict(),
            'train_loss': average_train_loss,
            'val_loss': average_val_loss,
            'best_loss': best_val_loss
        }
        save_checkpoint(checkpoint_state, filename=os.path.join(checkpoints_path, 'checkpoint_latest.pth.tar'))
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            save_checkpoint(checkpoint_state, filename=os.path.join(checkpoints_path, 'checkpoint_best.pth.tar'))
        save_checkpoint(checkpoint_state, filename=os.path.join(checkpoints_path, f'checkpoint_epoch_{epoch+1}.pth.tar'))

        all_latent_data = {'latent_codes': latent_codes.detach().cpu().numpy(), 'indices': torch.arange(latent_codes.size(0)).numpy()}
        np.save(os.path.join(latent_path, f'latent_codes_epoch_{epoch+1}.npy'), all_latent_data)

    np.save(os.path.join(checkpoints_path, 'loss_history.npy'), loss_history)



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
    # plt.ylim(0,1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, f'lossesandbeforeopt.png'))

    # plt.show()



def main():
    print("initializing device ..")
    device = initialize_device()
    print("loading config..")
    (config, data_path, checkpoints_path, latent_path, plots_path, tensorboard_path,
     batch_size, num_workers, num_epochs, learning_rate, latent_learning_rate, latent_dim, checkpoint_interval) = load_configurations()

    print("preparing datasets")
    train_loader, val_loader = prepare_datasets(config, data_path)

    print("initializing models and optimizers..")
    generator, latent_codes, optimizer_g, optimizer_l = initialize_models_and_optimizers(config, train_loader, device)

    print("loading checkpoints..")
    start_epoch, scheduler_g, scheduler_l, best_val_loss = load_checkpoints(generator, latent_codes, optimizer_g, optimizer_l, checkpoints_path, train_loader, config, device)

    print("start training and validation..")
    train_and_validate(generator, latent_codes, optimizer_g, optimizer_l, scheduler_g, scheduler_l, train_loader, val_loader, start_epoch, num_epochs, checkpoints_path, latent_path, device, best_val_loss)

    print("plotting losses..")
    plot_losses(os.path.join(checkpoints_path, 'loss_history.npy'),plots_path)

    
    
    print("done")

if __name__ == "__main__":
    main()
