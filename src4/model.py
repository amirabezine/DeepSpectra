import torch
import torch.nn as nn
import numpy as np
from torchinterp1d import interp1d

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, layers, activation_function='LeakyReLU'):
        super(Generator, self).__init__()
        modules = []
        input_dim = latent_dim
        self.latent_dim = latent_dim  # Save the latent dimension for later use
        for layer_dim in layers:
            modules.append(nn.Linear(input_dim, layer_dim))
            if activation_function == "LeakyReLU":
                modules.append(nn.LeakyReLU(0.2))
            else:
                try:
                    act_func = getattr(nn, activation_function)()
                except AttributeError:
                    raise ValueError(f"Activation function {activation_function} is not supported.")
                modules.append(act_func)
            input_dim = layer_dim
        modules.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*modules)
        self.apply(self.init_weights)  # Initialize weights

    def forward(self, z):
        return self.model(z)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DownsamplingLayer(nn.Module):
    def __init__(self, padding=5):
        super(DownsamplingLayer, self).__init__()
        self.padding = padding
        self.channels = [
            (4711, 4906, 0.05),
            (5647, 5875, 0.05),
            (6475, 6737, 0.05),
            (7583, 7885, 0.05),
            (15100, 17000, 0.2)
        ]

    def extend_wavelength_grid(self, observed_wavelength):
        extended_wavelengths = []
        for start, end, step in self.channels:
            pre_pad = np.arange(start - self.padding * step, start, step)
            post_pad = np.arange(end + step, end + (self.padding + 1) * step, step)
            channel = np.arange(start, end + step, step)
            extended_wavelengths.extend(pre_pad)
            extended_wavelengths.extend(channel)
            extended_wavelengths.extend(post_pad)
        extended_wavelengths = np.unique(np.array(extended_wavelengths))
        return torch.FloatTensor(extended_wavelengths)

    def forward(self, high_res_flux, high_res_wavelength, observed_wavelengths, device):
        high_res_wavelength = torch.FloatTensor(high_res_wavelength).to(device)
        
        # Ensure high_res_wavelength is sorted
        high_res_wavelength, sorted_indices = high_res_wavelength.sort()
        high_res_flux = high_res_flux[:, sorted_indices]

        # Extend the wavelength grid once for all batches
        extended_wavelength = self.extend_wavelength_grid(observed_wavelengths[0].cpu()).to(device)

        # Perform interpolation for the whole batch
        interpolated_fluxes = []
        for i in range(high_res_flux.size(0)):
            extended_interpolated_flux = interp1d(
                high_res_wavelength.unsqueeze(0),
                high_res_flux[i].unsqueeze(0),
                extended_wavelength.unsqueeze(0)
            )

            observed_interpolated_flux = interp1d(
                high_res_wavelength.unsqueeze(0),
                high_res_flux[i].unsqueeze(0),
                observed_wavelengths[i].unsqueeze(0)
            )

            interpolated_fluxes.append(observed_interpolated_flux.squeeze())

        return torch.stack(interpolated_fluxes)

class FullNetwork(nn.Module):
    def __init__(self, generator, high_res_wavelength, device):
        super(FullNetwork, self).__init__()
        self.generator = generator
        self.downsampling_layer = DownsamplingLayer()
        self.high_res_wavelength = high_res_wavelength
        self.device = device
    
    def forward(self, z, observed_wavelengths):
        high_res_flux = self.generator(z)  # Generate high-res flux from the generator
        downsampled_flux = self.downsampling_layer(high_res_flux, self.high_res_wavelength, observed_wavelengths, self.device)
        return downsampled_flux
