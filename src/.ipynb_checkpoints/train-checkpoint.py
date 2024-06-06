import torch
import torch.optim as optim
from tqdm import tqdm
from model import Generator
from dataset import APOGEEDataset
from checkpoint import save_checkpoint, load_checkpoint
from utils import get_config, resolve_path
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np

def weighted_mse_loss(input, target, weight):
    assert input.shape == target.shape == weight.shape, f'Shapes of input {input.shape}, target {target.shape}, and weight {weight.shape} must match'
    loss = torch.mean(weight * (input - target) ** 2)
    return loss

def validate_glo(generator, latent_code, dataloader, device):
    generator.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            flux = data['flux'].to(device)
            mask = data['flux_mask'].to(device)

            # flux_hat = generator(latent_code)
            # Expand latent_code to match the batch size
            flux_hat = generator(latent_code.expand(data['flux'].size(0), -1))

            weight = mask  
            loss = weighted_mse_loss(flux_hat, flux, weight)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_glo(generator, latent_code, train_loader, val_loader, config, device):
    latest_checkpoint_path = resolve_path(config['paths']['checkpoints']) + '/checkpoint_latest.pth.tar'
    best_checkpoint_path = resolve_path(config['paths']['checkpoints']) + '/checkpoint_best.pth.tar'
    latest_checkpoint = load_checkpoint(latest_checkpoint_path)
    best_checkpoint = load_checkpoint(best_checkpoint_path)

    optimizer = optim.Adam(list(generator.parameters()) + [latent_code], lr=config['training']['learning_rate'])
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if latest_checkpoint:
        generator.load_state_dict(latest_checkpoint['state_dict'])
        optimizer.load_state_dict(latest_checkpoint['optimizer'])
        start_epoch = latest_checkpoint['epoch']
        best_val_loss = latest_checkpoint['best_loss']
    
    generator.train()
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        total_train_loss = 0.0
        for data in tqdm(train_loader, total=len(train_loader)):
            flux = data['flux'].to(device)
            mask = data['flux_mask'].to(device)
    
            optimizer.zero_grad()
            # flux_hat = generator(latent_code)
            # print(f'Generated flux_hat shape: {flux_hat.shape}')
            # Expand latent_code to match the batch size
            flux_hat = generator(latent_code.expand(data['flux'].size(0), -1))


            weight = mask  
            loss = weighted_mse_loss(flux_hat, flux, weight)
            loss.backward()
            optimizer.step()
    
            total_train_loss += loss.item()
    
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Training Loss: {avg_train_loss:.4f}')
        
        avg_val_loss = validate_glo(generator, latent_code, val_loader, device)
        val_loss_history.append(avg_val_loss)
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Validation Loss: {avg_val_loss:.4f}')

        is_best = avg_val_loss < best_val_loss
        best_val_loss = min(avg_val_loss, best_val_loss)

        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
            'latent_code': latent_code,
            'optimizer': optimizer.state_dict(),
            'best_loss': best_val_loss
        }
        save_checkpoint(checkpoint_state, filename=latest_checkpoint_path)

        if is_best:
            save_checkpoint(checkpoint_state, filename=best_checkpoint_path)

        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(checkpoint_state, filename=resolve_path(config['paths']['checkpoints']) + f'/checkpoint_epoch_{epoch+1}.pth.tar')

    return generator, latent_code, train_loss_history, val_loss_history

def plot_loss_history(train_loss_history, val_loss_history, filename='loss.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    config = get_config()
    
    os.makedirs(resolve_path(config['paths']['checkpoints']), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    hdf5_path = resolve_path(config['paths']['hdf5_data'])
    num_workers = config['training']['num_workers']
    
    dataset = APOGEEDataset(hdf5_path, max_files=config['training']['max_files'])
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=config['training']['split_ratios'][1])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)

    latent_dim = config['training']['latent_dim']
    sample = next(iter(train_loader))
    flux_example = sample['flux'].to(device)
    wavelength_example = sample['wavelength'].to(device)
    output_dim = flux_example.size(1)

    generator_layers = config['model']['generator_layers']
    activation_function = getattr(torch.nn, config['model']['activation_function'])

    generator = Generator(latent_dim, output_dim, generator_layers, activation_function).to(device)

    latent_code = torch.randn(latent_dim, requires_grad=True, device=device)

    trained_generator, trained_latent_code, train_loss_history, val_loss_history = train_glo(generator, latent_code, train_loader, val_loader, config, device)

    test_latent_code = trained_latent_code.detach().cpu().numpy().flatten()
    generated_sample = trained_generator(torch.tensor(test_latent_code, device=device)).detach().cpu().numpy().flatten()

    # plot_spectrum(wavelength_example.cpu().numpy(), generated_sample, original_flux=flux_example[0].cpu().numpy(), filename='spectrum.png')
    plot_loss_history(train_loss_history, val_loss_history, filename='loss.png')
