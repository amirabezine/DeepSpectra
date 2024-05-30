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
import h5py
import matplotlib.pyplot as plt

def weighted_mse_loss(input, target, weight):
    assert input.shape == target.shape == weight.shape, f'Shapes of input {input.shape}, target {target.shape}, and weight {weight.shape} must match'
    loss = torch.mean(weight * (input - target) ** 2)
    return loss

def validate_glo(generator, latent_dim, dataloader, device):
    generator.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            flux = data['flux'].to(device)
            mask = data['flux_mask'].to(device)
            batch_size = flux.size(0)

            # Generate latent code for the batch
            latent_code = torch.randn(batch_size, latent_dim, requires_grad=False, device=device)
            flux_hat = generator(latent_code)

            weight = mask  
            loss = weighted_mse_loss(flux_hat, flux, weight)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_glo(generator, train_loader, val_loader, config, device, save_latent_interval=10):
    latest_checkpoint_path = resolve_path(config['paths']['checkpoints']) + '/checkpoint_latest.pth.tar'
    best_checkpoint_path = resolve_path(config['paths']['checkpoints']) + '/checkpoint_best.pth.tar'
    latent_path = resolve_path(config['paths']['latent'])
    os.makedirs(latent_path, exist_ok=True)

    latest_checkpoint = load_checkpoint(latest_checkpoint_path)
    best_checkpoint = load_checkpoint(best_checkpoint_path)

    optimizer_weights = optim.Adam(generator.parameters(), lr=config['training']['learning_rate'])
    scheduler_weights = optim.lr_scheduler.StepLR(optimizer_weights, step_size=config['training']['scheduler_step_size'], gamma=config['training']['scheduler_gamma'])

    start_epoch = 0
    best_val_loss = float('inf')
    
    if latest_checkpoint:
        generator.load_state_dict(latest_checkpoint['state_dict'])
        optimizer_weights.load_state_dict(latest_checkpoint['optimizer'])
        start_epoch = latest_checkpoint['epoch']
        best_val_loss = latest_checkpoint['best_loss']
    
    generator.train()
    
    train_loss_history = []
    val_loss_history = []

    for epoch in range(start_epoch, config['training']['num_epochs']):
        total_train_loss = 0.0
        for batch_idx, data in enumerate(tqdm(train_loader, total=len(train_loader))):
            flux = data['flux'].to(device)
            mask = data['flux_mask'].to(device)
            spectrum_index = data['index']
            batch_size = flux.size(0)

            # Generate latent code for the batch
            latent_code = torch.randn(batch_size, config['training']['latent_dim'], requires_grad=True, device=device)
            optimizer_latent = optim.Adam([latent_code], lr=config['training']['latent_learning_rate'])

            # Step 1: Optimize generator weights
            generator.train()
            optimizer_weights.zero_grad()
            flux_hat = generator(latent_code)
            weight = mask  
            loss = weighted_mse_loss(flux_hat, flux, weight)
            loss.backward()
            optimizer_weights.step()
            
            # Step 2: Freeze weights and optimize latent vectors
            generator.eval()
            optimizer_latent.zero_grad()
            flux_hat = generator(latent_code)
            loss = weighted_mse_loss(flux_hat, flux, weight)
            loss.backward()
            optimizer_latent.step()

            total_train_loss += loss.item()

            # Save the optimized latent codes every save_latent_interval epochs
            if (epoch + 1) % save_latent_interval == 0:
                for i in range(batch_size):
                    latent_save_path = os.path.join(latent_path, f'latent_epoch_{epoch+1}_batch_{batch_idx+1}_index_{spectrum_index[i].item()}.pt')
                    torch.save({
                        'index': spectrum_index[i].item(),
                        'latent_code': latent_code[i].detach().cpu()
                    }, latent_save_path)
    
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Training Loss: {avg_train_loss:.4f}')
        
        avg_val_loss = validate_glo(generator, config['training']['latent_dim'], val_loader, device)
        val_loss_history.append(avg_val_loss)
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Validation Loss: {avg_val_loss:.4f}')

        is_best = avg_val_loss < best_val_loss
        best_val_loss = min(avg_val_loss, best_val_loss)

        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
            'optimizer': optimizer_weights.state_dict(),
            'best_loss': best_val_loss
        }
        save_checkpoint(checkpoint_state, filename=latest_checkpoint_path)

        if is_best:
            save_checkpoint(checkpoint_state, filename=best_checkpoint_path)

        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(checkpoint_state, filename=resolve_path(config['paths']['checkpoints']) + f'/checkpoint_epoch_{epoch+1}.pth.tar')

        scheduler_weights.step()

    return generator, train_loss_history, val_loss_history

def plot_loss_history(train_loss_history, val_loss_history, config, filename='loss.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()

    # Get the plots path from config and save the plot there
    plots_path = resolve_path(config['paths']['plots'])
    os.makedirs(plots_path, exist_ok=True)
    plt.savefig(os.path.join(plots_path, filename))
    plt.show()

def plot_latent_evolution(latent_path, batch_idx, spectrum_idx, num_epochs, config):
    latent_vectors = []
    epochs = range(1, num_epochs+1)
    
    for epoch in epochs:
        latent_file = os.path.join(latent_path, f'latent_epoch_{epoch}_batch_{batch_idx + 1}_index_{spectrum_idx}.pt')
        if os.path.exists(latent_file):
            latent_data = torch.load(latent_file)
            latent_vectors.append(latent_data['latent_code'].cpu().numpy())
    
    plt.figure(figsize=(10, 6))
    for i, latent_vector in enumerate(latent_vectors):
        plt.plot(latent_vector, label=f'Epoch {epochs[i]}')
    
    plt.xlabel('Latent Dimension')
    plt.ylabel('Value')
    plt.title(f'Latent Vector Evolution for Spectrum Index {spectrum_idx} in Batch {batch_idx + 1}')
    plt.legend()
    
    plots_path = resolve_path(config['paths']['plots'])
    os.makedirs(plots_path, exist_ok=True)
    plt.savefig(os.path.join(plots_path, f'latent_evolution_spectrum_{spectrum_idx}_batch_{batch_idx + 1}.png'))
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

    trained_generator, train_loss_history, val_loss_history = train_glo(generator, train_loader, val_loader, config, device)

    plot_loss_history(train_loss_history, val_loss_history, config, filename='loss.png')

    # Plot latent evolution for batch index 2 and spectrum number 1
    plot_latent_evolution(config['paths']['latent'], batch_idx=1, spectrum_idx=0, num_epochs=config['training']['num_epochs'], config=config)
