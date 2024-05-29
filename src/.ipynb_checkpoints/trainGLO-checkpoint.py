import torch
import torch.optim as optim
from tqdm import tqdm
from model import Generator  # Ensure this imports the correct Generator class definition
from dataset import APOGEEDataset, get_dataloaders
from checkpoint import save_checkpoint, load_checkpoint
from utils import get_config, resolve_path
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def weighted_mse_loss(input, target, weight):
    assert input.shape == target.shape == weight.shape, f'Shapes of input {input.shape}, target {target.shape}, and weight {weight.shape} must match'
    loss = torch.mean(weight * (input - target) ** 2)
    return loss

def validate_glo(generator, latent_codes, dataloader, device):
    generator.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            flux = data['flux'].to(device)
            mask = data['flux_mask'].to(device)

            # Retrieve the corresponding latent codes for the current batch
            batch_latent_codes = latent_codes[data['index']].to(device)
            flux_hat = generator(batch_latent_codes)

            weight = mask  
            loss = weighted_mse_loss(flux_hat, flux, weight)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_glo(generator, latent_codes, train_loader, val_loader, config, device):
    latest_checkpoint_path = resolve_path(config['paths']['checkpoints']) + '/checkpoint_latest.pth.tar'
    best_checkpoint_path = resolve_path(config['paths']['checkpoints']) + '/checkpoint_best.pth.tar'
    latest_checkpoint = load_checkpoint(latest_checkpoint_path)
    best_checkpoint = load_checkpoint(best_checkpoint_path)

    # Initialize the optimizer with both generator parameters and latent codes
    optimizer = optim.Adam(list(generator.parameters()) + [latent_codes], lr=config['training']['learning_rate'])
    
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
        for i, data in enumerate(tqdm(train_loader, total=len(train_loader))):
            flux = data['flux'].to(device)
            mask = data['flux_mask'].to(device)

            optimizer.zero_grad()
            
            # Retrieve and use the corresponding latent codes for the current batch
            batch_latent_codes = latent_codes[data['index']].to(device)
            flux_hat = generator(batch_latent_codes)

            weight = mask  
            loss = weighted_mse_loss(flux_hat, flux, weight)
            loss.backward()
            optimizer.step()

            # Project latent codes onto the unit sphere
            with torch.no_grad():
                latent_codes[data['index']] = latent_codes[data['index']] / torch.norm(latent_codes[data['index']], dim=1, keepdim=True)

            total_train_loss += loss.item()
    
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Training Loss: {avg_train_loss:.4f}')
        
        avg_val_loss = validate_glo(generator, latent_codes, val_loader, device)
        val_loss_history.append(avg_val_loss)
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Validation Loss: {avg_val_loss:.4f}')

        is_best = avg_val_loss < best_val_loss
        best_val_loss = min(avg_val_loss, best_val_loss)

        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
            'latent_codes': latent_codes,
            'optimizer': optimizer.state_dict(),
            'best_loss': best_val_loss
        }
        save_checkpoint(checkpoint_state, filename=latest_checkpoint_path)

        if is_best:
            save_checkpoint(checkpoint_state, filename=best_checkpoint_path)

        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(checkpoint_state, filename=resolve_path(config['paths']['checkpoints']) + f'/checkpoint_epoch_{epoch+1}.pth.tar')

    return generator, latent_codes, train_loss_history, val_loss_history

def plot_loss_history(train_loss_history, val_loss_history, filename='lossGLO.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.savefig(filename)
    plt.close() 
    # plt.show()

def plot_latent_space(latent_codes, epoch, filename='latent_spaceGLO.png'):
    latent_codes_np = latent_codes.detach().cpu().numpy()
    
    tsne = TSNE(n_components=2, random_state=42)
    latent_codes_2d = tsne.fit_transform(latent_codes_np)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(latent_codes_2d[:, 0], latent_codes_2d[:, 1], c='blue', s=5)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(f'Latent Space Distribution at Epoch {epoch}')
    plt.savefig(filename)
    plt.close() 
    # plt.show()

if __name__ == "__main__":
    config = get_config()
    
    os.makedirs(resolve_path(config['paths']['checkpoints']), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    hdf5_path = resolve_path(config['paths']['hdf5_data'])
    num_workers = config['training']['num_workers']
    
    train_loader, val_loader, test_loader = get_dataloaders(hdf5_path, config['training']['batch_size'], num_workers, config['training']['split_ratios'])

    latent_dim = config['training']['latent_dim']
    num_samples = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)

    # Initialize latent codes for all samples in the dataset
    latent_codes = torch.randn(num_samples, latent_dim, requires_grad=True, device=device)

    generator_layers = config['model']['generator_layers']
    output_dim = config['model']['output_dim']
    activation_function = getattr(torch.nn, config['model']['activation_function'])

    # Initialize the generator with the provided parameters
    generator = Generator(latent_dim, output_dim, generator_layers, activation_function).to(device)

    trained_generator, trained_latent_codes, train_loss_history, val_loss_history = train_glo(generator, latent_codes, train_loader, val_loader, config, device)

    # Plot the latent space distribution for each epoch
    for epoch in range(config['training']['num_epochs']):
        plot_latent_space(trained_latent_codes, epoch, filename=f'latent_space_epoch_{epoch+1}.png')

    plot_loss_history(train_loss_history, val_loss_history, filename='lossGLO.png')
