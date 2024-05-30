import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA
from model import Generator
from dataset import get_dataloaders

def weighted_mse_loss(predicted, target, weight):
    return torch.mean(weight * (predicted - target) ** 2)

def log_latent_vectors(writer, latent_vectors, epoch, log_interval=10):
    if epoch % log_interval == 0:
        with torch.no_grad():
            latent_vectors_sample = latent_vectors[:500].cpu().numpy()  # Log a subset of latent vectors
            pca = PCA(n_components=2)
            latent_vectors_pca = pca.fit_transform(latent_vectors_sample)
            for i, vector in enumerate(latent_vectors_pca):
                writer.add_scalars(f'Latent_vector_{i}', {'x': vector[0], 'y': vector[1]}, epoch)

def train_glo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        config['paths']['hdf5_data'],
        config['training']['batch_size'],
        config['training']['num_workers'],
        config['training']['split_ratios']
    )
    
    # Initialize model, optimizer, and tensorboard writer
    generator = Generator(
        latent_dim=config['training']['latent_dim'],
        output_dim=config['model']['output_dim'],
        generator_layers=config['model']['generator_layers'],
        activation_function=config['model']['activation_function']
    ).to(device)
    
    optimizer = optim.Adam(generator.parameters(), lr=config['training']['learning_rate'])
    writer = SummaryWriter(config['paths']['tensorboard'])
    
    # Initialize latent vectors
    latent_vectors = torch.randn(len(train_loader.dataset), config['training']['latent_dim'], requires_grad=True, device=device)
    latent_optimizer = optim.Adam([latent_vectors], lr=config['training']['learning_rate'])
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        generator.train()
        train_loss = 0
        
        for batch in train_loader:
            indices = batch['index'].to(device)
            flux = batch['flux'].to(device)
            sigma = batch['sigma'].to(device)
            flux_mask = batch['flux_mask'].to(device)
            weight = sigma * flux_mask
            
            # Generate reconstructed flux
            latent_batch = latent_vectors[indices]
            reconstructed_flux = generator(latent_batch)
            
            # Calculate loss
            loss = weighted_mse_loss(reconstructed_flux, flux, weight)
            optimizer.zero_grad()
            latent_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            latent_optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validation loop
        generator.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                indices = batch['index'].to(device)
                flux = batch['flux'].to(device)
                sigma = batch['sigma'].to(device)
                flux_mask = batch['flux_mask'].to(device)
                weight = sigma * flux_mask
                
                latent_batch = latent_vectors[indices]
                reconstructed_flux = generator(latent_batch)
                
                loss = weighted_mse_loss(reconstructed_flux, flux, weight)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Log latent vectors for tracking
        log_latent_vectors(writer, latent_vectors, epoch)
        
        # Save checkpoints
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            torch.save(generator.state_dict(), os.path.join(config['paths']['checkpoints'], f'generator_epoch_{epoch+1}.pth'))
            torch.save(latent_vectors, os.path.join(config['paths']['checkpoints'], f'latent_vectors_epoch_{epoch+1}.pth'))
        
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    writer.close()



if __name__ == "__main__":
    train_glo(config)