import torch
import torch.optim as optim
from tqdm import tqdm
from model import Generator
from dataset import get_dataloaders
import os
import yaml
from torch.utils.tensorboard import SummaryWriter



def weighted_mse_loss(output, target, weight):
    return torch.mean(weight * (output - target) ** 2)

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def train_glo(config):
    writer = SummaryWriter(log_dir=config['paths']['tensorboard'])
    
    generator = Generator(
        config['training']['latent_dim'],
        config['model']['output_dim'],
        config['model']['generator_layers'],
        getattr(torch.nn, config['model']['activation_function'])
    )
    
    train_loader, val_loader, test_loader = get_dataloaders(
        config['paths']['hdf5_data'],
        config['training']['batch_size'],
        config['training']['num_workers'],
        config['training']['split_ratios']
    )

    generator_optimizer = optim.Adam(generator.parameters(), lr=config['training']['learning_rate'])
    generator.train()
    
    latent_codes = torch.randn(len(train_loader.dataset), config['training']['latent_dim'], requires_grad=True)
    latent_optimizer = optim.Adam([latent_codes], lr=config['training']['learning_rate'])

    loss_history = []

    best_loss = float('inf')

    for epoch in range(config['training']['num_epochs']):
        total_loss = 0.0
        for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
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
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        writer.add_scalar('Training Loss', avg_loss, epoch)
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': generator.state_dict(),
                'optimizer': generator_optimizer.state_dict(),
                'loss': avg_loss,
                'latent_codes': latent_codes
            }, filename=os.path.join(config['paths']['checkpoints'], f'checkpoint_epoch_{epoch+1}.pth.tar'))

    writer.close()
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

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    
    trained_generator, loss_history = train_glo(config)
    plot_loss_history(loss_history)




