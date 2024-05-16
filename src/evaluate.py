import torch
from dataset import get_dataloaders
from model import Generator
import matplotlib.pyplot as plt

def plot_spectrum(wavelength, generated_flux, original_flux=None):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, generated_flux, label='Generated Flux', color='blue')
    if original_flux is not None:
        plt.plot(wavelength, original_flux, label='Original Flux', color='red', alpha=0.5)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Flux vs. Wavelength')
    plt.legend()
    plt.show()

def evaluate(config):
    generator = Generator(
        config['training']['latent_dim'],
        config['model']['output_dim'],
        config['model']['generator_layers'],
        getattr(torch.nn, config['model']['activation_function'])
    )
    
    _, _, test_loader = get_dataloaders(
        config['paths']['hdf5_data'],
        config['training']['batch_size'],
        config['training']['num_workers'],
        config['training']['split_ratios']
    )

    test_latent_code = torch.randn(1, config['training']['latent_dim'])
    generated_sample = generator(test_latent_code).detach().numpy().flatten()

    sample = next(iter(test_loader))
    flux_example = sample['flux'][0]
    wavelength_example = sample['wavelength'][0]

    plot_spectrum(wavelength_example.numpy(), generated_sample, original_flux=flux_example.numpy())

if __name__ == "__main__":
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate(config)
