import torch
from config import Config
from device import initialize_device
from data import DataPreparation
from optimization import OptimizerInitializer
from train import Trainer
from checkpoint import CheckpointManager
from plotting import Plotter
from model import SpectrumGenerator

def main():
    print("Initializing device...")
    device = initialize_device()

    print("Loading configuration...")
    config = Config()

    print("Preparing datasets...")
    data_prep = DataPreparation(config)
    train_loader, val_loader = data_prep.prepare_datasets()

    print("Initializing model and optimizers...")
    # Correctly define the high-resolution wavelength grid
    wavelength_grid = [
        (15050, 15850, 0.2),
        (15870, 16440, 0.2),
        (16475, 17100, 0.2),
        (4700, 4930, 0.05),
        (5630, 5880, 0.05),
        (6420, 6800, 0.05),
        (7500, 7920, 0.05)
    ]
    high_res_wavelength_grid = torch.cat([torch.arange(start, end + step, step) for start, end, step in wavelength_grid])

    # Initialize the SpectrumGenerator with correct parameters
    spectrum_generator = SpectrumGenerator(config.latent_dim, high_res_wavelength_grid.shape[0], config.generator_layers, config.activation_function).to(device)

    # Initialize latent codes with correct dimensions if needed
    latent_codes = torch.randn(1, config.latent_dim, device=device)  # Initialize with a single random vector
    dict_latent_codes = {}
    optimizer_initializer = OptimizerInitializer(config, spectrum_generator, latent_codes)

    optimizer_g, optimizer_l = optimizer_initializer.initialize_optimizers()

    print("Loading checkpoints...")
    checkpoint_manager = CheckpointManager(config, spectrum_generator, optimizer_g, optimizer_l, device, train_loader)
    start_epoch, scheduler_g, scheduler_l, best_val_loss, latent_codes, dict_latent_codes, optimizer_l = checkpoint_manager.load_checkpoints()

    print("Starting training and validation...")
    trainer = Trainer(config, spectrum_generator, latent_codes, dict_latent_codes, optimizer_g, optimizer_l, scheduler_g, scheduler_l, train_loader, val_loader, start_epoch, device, best_val_loss)
    latent_codes, dict_latent_codes = trainer.train_and_validate()

    print("Plotting losses...")
    plotter = Plotter(config)
    plotter.plot_losses()

    print("Done.")

if __name__ == "__main__":
    main()
