import torch
from config import Config
from device import initialize_device
from data import DataPreparation
from optimization import OptimizerInitializer
from train import Trainer
from checkpoint import CheckpointManager
from plotting import Plotter
from model import Generator

def main():
    print("Initializing device...")
    device = initialize_device()

    print("Loading configuration...")
    config = Config()

    print("Preparing datasets...")
    data_prep = DataPreparation(config)
    train_loader, val_loader = data_prep.prepare_datasets()

    print("Initializing model and optimizers...")
    generator = Generator(config.latent_dim, config.output_dim, config.generator_layers, config.activation_function).to(device)
    latent_codes = torch.randn(1, config.latent_dim, device=device)  # Initialize with a single random vector
    dict_latent_codes = {}
    optimizer_initializer = OptimizerInitializer(config, generator, latent_codes)
    optimizer_g, optimizer_l = optimizer_initializer.initialize_optimizers()

    print("Loading checkpoints...")
    checkpoint_manager = CheckpointManager(config, generator, optimizer_g, optimizer_l, device, train_loader)
    start_epoch, scheduler_g, scheduler_l, best_val_loss, latent_codes, dict_latent_codes, optimizer_l = checkpoint_manager.load_checkpoints()

    print("Starting training and validation...")
    trainer = Trainer(config, generator, latent_codes, dict_latent_codes, optimizer_g, optimizer_l, scheduler_g, scheduler_l, train_loader, val_loader, start_epoch, device, best_val_loss)
    latent_codes, dict_latent_codes = trainer.train_and_validate()

    print("Plotting losses...")
    plotter = Plotter(config)
    plotter.plot_losses()

    print("Done.")

if __name__ == "__main__":
    main()
