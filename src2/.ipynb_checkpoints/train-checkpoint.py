import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from ignite.engine import Events, Engine
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
import matplotlib.pyplot as plt
from model import Generator
from dataset import APOGEEDataset
from checkpoint import save_checkpoint, load_checkpoint
from utils import get_config, resolve_path
import logging
import os

# Setup logger
def setup_logger(level='DEBUG'):
    logger = logging.getLogger(__name__)
    hdlr = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s|%(name)s|%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(level)
    return logger

logger = setup_logger()

# Get DataLoader
def get_dataloader(config):
    data_transforms = transforms.Compose([transforms.ToTensor()])
    hdf5_path = resolve_path(config['paths']['hdf5_data'])
    dataset = APOGEEDataset(hdf5_path, max_files=config['training']['max_files'])
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=config['training']['split_ratios'][1])

    train_dataset = Subset(dataset, list(train_indices))
    val_dataset = Subset(dataset, list(val_indices))

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])
    return train_loader, val_loader

# Get activation function
def get_activation(activation_name):
    if activation_name == "LeakyReLU":
        return lambda: nn.LeakyReLU(0.2)
    elif activation_name == "ReLU":
        return lambda: nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

# Define Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, layers, activation_factory):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, layers[0]),
            activation_factory(),
            nn.Linear(layers[0], layers[1]),
            activation_factory(),
            nn.Linear(layers[1], output_dim)
        )

    def forward(self, z):
        return self.model(z)

# Create model
def create_model(config):
    activation_factory = get_activation(config['model']['activation_function'])
    generator = Generator(config['training']['latent_dim'], config['model']['output_dim'],
                          config['model']['generator_layers'], activation_factory)
    return generator

# Define weighted MSE loss function
def weighted_mse_loss(y_pred, y, weight):
    loss = torch.mean(weight * (y_pred - y) ** 2)
    return loss

# Define custom weighted MSE loss metric
class WeightedMSELoss(Metric):
    def __init__(self, output_transform=lambda x: x, device=torch.device('cpu')):
        super(WeightedMSELoss, self).__init__(output_transform=output_transform, device=device)
        self.device = device

    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y, weight = output
        loss = weighted_mse_loss(y_pred, y, weight)
        self._sum += loss.item() * y.shape[0]
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('WeightedMSELoss must have at least one example before it can be computed.')
        return self._sum / self._num_examples

# Create trainer and evaluator
def create_trainer_and_evaluator(model, optimizer, latent_vectors, device, train_loader):
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        z, x, y, mask, sigma = prepare_batch(batch, latent_vectors, device=device)
        y_pred = model(z)
        weight = mask * sigma
        loss = weighted_mse_loss(y_pred, y, weight)
        loss.backward()
        # Print gradient norms
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # print(f"Gradient norm: {total_norm}")
        optimizer.step()
        # print(f"Train step: y_pred={y_pred.mean().item()}, y={y.mean().item()}, weight={weight.mean().item()}, loss={loss.item()}")
        return loss.item()

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            z, x, y, mask, sigma = prepare_batch(batch, latent_vectors, device=device)
            y_pred = model(z)
            weight = mask * sigma
            loss = weighted_mse_loss(y_pred, y, weight)
            # print(f"Validation step: y_pred={y_pred.mean().item()}, y={y.mean().item()}, weight={weight.mean().item()}, loss={loss.item()}")
            return y_pred, y, weight

    trainer = Engine(train_step)
    evaluator = Engine(validation_step)

    WeightedMSELoss(device=device).attach(evaluator, 'loss')

    return trainer, evaluator

# Prepare batch function
def prepare_batch(batch, latent_vectors, device=None, non_blocking=False):
    index = batch['index']
    z = latent_vectors[index].to(device)
    x = batch['wavelength'].to(device, non_blocking=non_blocking)
    y = batch['flux'].to(device, non_blocking=non_blocking)
    mask = batch['flux_mask'].to(device, non_blocking=non_blocking)
    sigma = batch['sigma'].to(device, non_blocking=non_blocking)
    return z, x, y, mask, sigma

# Save plots function
def save_plots(losses, latent_snapshots, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    # Plot and save losses
    plt.figure()
    plt.plot(losses, label='Validation Loss')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'validation_loss.png'))
    plt.close()

    # Plot and save latent vector norms over time
    norms = [torch.norm(snapshot, dim=1).mean().item() for snapshot in latent_snapshots]
    plt.figure()
    plt.plot(norms, label='Mean Latent Norm')
    plt.title('Mean Latent Vector Norm Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'latent_norms.png'))
    plt.close()

# Main function
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_loader, val_loader = get_dataloader(config)
    model = create_model(config).to(device)
    latent_vectors = torch.randn((len(train_loader.dataset), config['training']['latent_dim']), requires_grad=True, device=device)
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': latent_vectors, 'lr': config['training']['latent_learning_rate']}
    ], lr=config['training']['learning_rate'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    trainer, evaluator = create_trainer_and_evaluator(model, optimizer, latent_vectors, device, train_loader)

    losses = []
    latent_snapshots = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        avg_loss = evaluator.state.metrics['loss']
        losses.append(avg_loss)
        latent_snapshots.append(latent_vectors.clone().detach())
        logger.info(f"Validation Results - Epoch: {engine.state.epoch} Avg loss: {avg_loss:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_scheduler(engine):
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model(engine):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': engine.state.epoch,
            'latent_vectors': latent_vectors.cpu().detach().numpy()
        }
        save_checkpoint(checkpoint, filename=resolve_path(config['paths']['checkpoints']) + f'/checkpoint_epoch_{engine.state.epoch}.pth.tar')

    trainer.run(train_loader, max_epochs=config['training']['num_epochs'])

    save_plots(losses, latent_snapshots, config['paths']['plots'])

if __name__ == '__main__':
    config = get_config()
    main(config)
