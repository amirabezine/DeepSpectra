import torch
import torch.optim as optim
from tqdm import tqdm
from model import Generator
from dataset import APOGEEDataset
from checkpoint import save_checkpoint, load_checkpoint
from utils import get_config, resolve_path
from torch.utils.data import DataLoader
import os

def weighted_mse_loss(output, target, weight):
    # print(f'Output shape: {output.shape}, Target shape: {target.shape}')  # Debugging line
    return torch.mean(weight * (output - target) ** 2)

def train_glo(generator, dataloader, latent_dim, config):
    # Load checkpoint if available
    latest_checkpoint_path = resolve_path(config['paths']['checkpoints']) + '/checkpoint_latest.pth.tar'
    best_checkpoint_path = resolve_path(config['paths']['checkpoints']) + '/checkpoint_best.pth.tar'
    latest_checkpoint = load_checkpoint(latest_checkpoint_path)
    best_checkpoint = load_checkpoint(best_checkpoint_path)

    generator_optimizer = optim.Adam(generator.parameters(), lr=config['training']['learning_rate'])
    
    start_epoch = 0
    best_loss = float('inf')
    
    if latest_checkpoint:
        generator.load_state_dict(latest_checkpoint['state_dict'])
        generator_optimizer.load_state_dict(latest_checkpoint['optimizer'])
        start_epoch = latest_checkpoint['epoch']
        best_loss = latest_checkpoint['best_loss']
    
    generator.train()
    
    latent_codes = torch.randn(len(dataloader.dataset), latent_dim, requires_grad=True)
    latent_optimizer = optim.Adam([latent_codes], lr=config['training']['learning_rate'])
    
    loss_history = []

    for epoch in range(start_epoch, config['training']['num_epochs']):
        total_loss = 0.0
        for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            flux = data['flux']
            sigma = data['sigma']
            mask = data['flux_mask']

            z = latent_codes[idx].unsqueeze(0)
            flux_hat = generator(z)
            weight = mask / (sigma ** 2 + 1e-10)
            loss = weighted_mse_loss(flux_hat, flux, weight)
            
            generator_optimizer.zero_grad()
            latent_optimizer.zero_grad()
            loss.backward()
            generator_optimizer.step()
            latent_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Loss: {avg_loss:.4f}')

        is_best = avg_loss < best_loss
        best_loss = min(avg_loss, best_loss)

        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
            'optimizer': generator_optimizer.state_dict(),
            'best_loss': best_loss,
            'latent_codes': latent_codes
        }
        save_checkpoint(checkpoint_state, filename=latest_checkpoint_path)

        if is_best:
            save_checkpoint(checkpoint_state, filename=best_checkpoint_path)

        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(checkpoint_state, filename=resolve_path(config['paths']['checkpoints']) + f'/checkpoint_epoch_{epoch+1}.pth.tar')

    return generator, loss_history

def plot_loss_history(loss_history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.show()

def plot_spectrum(wavelength, generated_flux, original_flux=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, generated_flux, label='Generated Flux', color='blue')
    if original_flux is not None:
        plt.plot(wavelength, original_flux, label='Original Flux', color='red', alpha=0.5)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Flux vs. Wavelength')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    config = get_config()
    
    os.makedirs(resolve_path(config['paths']['checkpoints']), exist_ok=True)
    
    hdf5_path = resolve_path(config['paths']['hdf5_data'])
    num_workers = config['training']['num_workers']
    dataset = APOGEEDataset(hdf5_path, max_files=config['training']['max_files'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    latent_dim = config['training']['latent_dim']
    sample = next(iter(dataloader))
    flux_example = sample['flux']
    wavelength_example = sample['wavelength']
    output_dim = flux_example.size(0)

    generator_layers = config['model']['generator_layers']
    activation_function = getattr(torch.nn, config['model']['activation_function'])

    generator = Generator(latent_dim, output_dim, generator_layers, activation_function)
    
    trained_generator, loss_history = train_glo(generator, dataloader, latent_dim, config)

    test_latent_code = torch.randn(1, latent_dim)
    generated_sample = trained_generator(test_latent_code).detach().numpy().flatten()

    plot_spectrum(wavelength_example.numpy(), generated_sample, original_flux=flux_example.numpy())
    plot_loss_history(loss_history)
