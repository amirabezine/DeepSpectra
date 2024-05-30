import torch
import torch.optim as optim
from tqdm import tqdm
from model import Generator
from dataset import APOGEEDataset
from checkpoint import save_checkpoint, load_checkpoint
from utils import get_config, resolve_path
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def train_glo(generator, train_loader, val_loader, config, device):
    latent_codes = {idx: torch.randn(config['training']['latent_dim'], requires_grad=True, device=device)
                    for idx in range(len(train_loader.dataset))}
    optimizer = optim.Adam(list(generator.parameters()) + list(latent_codes.values()), lr=config['training']['learning_rate'])
    train_loss_history, val_loss_history = [], []
    latent_code_snapshots = []

    for epoch in range(config['training']['num_epochs']):
        generator.train()
        total_train_loss = 0.0
        for data in tqdm(train_loader, total=len(train_loader)):
            flux, mask = data['flux'].to(device), data['flux_mask'].to(device)
            idxs = data['index']

            optimizer.zero_grad()
            batch_latent_codes = torch.stack([latent_codes[idx.item()] for idx in idxs])
            flux_hat = generator(batch_latent_codes)

            loss = weighted_mse_loss(flux_hat, flux, mask)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        generator.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for data in val_loader:
                flux, mask = data['flux'].to(device), data['flux_mask'].to(device)
                idxs = data['index']
                batch_latent_codes = torch.stack([latent_codes[idx.item()] for idx in idxs])
                flux_hat = generator(batch_latent_codes)
                val_loss = weighted_mse_loss(flux_hat, flux, mask)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        if epoch % 5 == 0:  # Adjust the interval as needed
            latent_snapshot = {k: v.detach().cpu().numpy() for k, v in latent_codes.items()}
            latent_code_snapshots.append((epoch, latent_snapshot))

        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    return generator, latent_codes, train_loss_history, val_loss_history, latent_code_snapshots

def plot_loss_history(train_loss_history, val_loss_history, config):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.savefig(resolve_path(config['paths']['plots']) + 'loss_history.png')
    plt.close()

def plot_latent_codes(latent_code_snapshots, config):
    pca = PCA(n_components=2)
    fig, ax = plt.subplots(figsize=(8, 6))

    for epoch, snapshot in latent_code_snapshots:
        all_codes = np.array(list(snapshot.values()))
        pca_result = pca.fit_transform(all_codes)
        ax.scatter(pca_result[:, 0], pca_result[:, 1], label=f'Epoch {epoch}')

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('PCA of Latent Codes over Epochs')
    ax.legend()
    plt.savefig(resolve_path(config['paths']['plots']) + 'latent_code_evolution.png')
    plt.close()

if __name__ == "__main__":
    config = get_config()

    os.makedirs(resolve_path(config['paths']['checkpoints']), exist_ok=True)
    os.makedirs(resolve_path(config['paths']['plots']), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    hdf5_path = resolve_path(config['paths']['hdf5_data'])
    dataset = APOGEEDataset(hdf5_path, max_files=config['training']['max_files'])
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=config['training']['split_ratios'][1])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])

    generator_layers = config['model']['generator_layers']
    activation_function = getattr(torch.nn, config['model']['activation_function'])
    output_dim = next(iter(train_loader))['flux'].size(1)

    generator = Generator(config['training']['latent_dim'], output_dim, generator_layers, activation_function).to(device)

    trained_generator, latent_codes, train_loss_history, val_loss_history, latent_code_snapshots = train_glo(generator, train_loader, val_loader, config, device)

    plot_loss_history(train_loss_history, val_loss_history, config)
    plot_latent_codes(latent_code_snapshots, config)
