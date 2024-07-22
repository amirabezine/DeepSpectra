import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.extended_wavelength = self.create_extended_wavelength_grid()

    def create_extended_wavelength_grid(self):
        extended_wavelengths = []
        for start, end, step in self.channels:
            pre_pad = np.arange(start - self.padding * step, start, step)
            post_pad = np.arange(end + step, end + (self.padding + 1) * step, step)
            channel = np.arange(start, end + step, step)
            extended_wavelengths.extend(pre_pad)
            extended_wavelengths.extend(channel)
            extended_wavelengths.extend(post_pad)
        return torch.FloatTensor(np.unique(extended_wavelengths))

    def forward(self, high_res_flux, high_res_wavelength, observed_wavelengths, device):
        print("Entering DownsamplingLayer forward")
        try:
            high_res_flux = high_res_flux.to(device)
            high_res_wavelength = high_res_wavelength.to(device)
            observed_wavelengths = observed_wavelengths.to(device)
            extended_wavelength = self.extended_wavelength.to(device)
            
            print(f"high_res_flux: shape={high_res_flux.shape}, dtype={high_res_flux.dtype}")
            print(f"high_res_wavelength: shape={high_res_wavelength.shape}, dtype={high_res_wavelength.dtype}")
            print(f"observed_wavelengths: shape={observed_wavelengths.shape}, dtype={observed_wavelengths.dtype}")
            print(f"extended_wavelength: shape={extended_wavelength.shape}, dtype={extended_wavelength.dtype}")

            # Check for NaNs or infinities
            if torch.isnan(high_res_flux).any() or torch.isinf(high_res_flux).any():
                print("NaN or Inf detected in high_res_flux")
            if torch.isnan(high_res_wavelength).any() or torch.isinf(high_res_wavelength).any():
                print("NaN or Inf detected in high_res_wavelength")

            # Interpolate to extended wavelength grid
            extended_flux = interp1d(high_res_wavelength, high_res_flux, extended_wavelength)
            print("Extended flux interpolated")

            # Check for NaNs or infinities after first interpolation
            if torch.isnan(extended_flux).any() or torch.isinf(extended_flux).any():
                print("NaN or Inf detected in extended_flux")

            # Interpolate to observed wavelengths
            observed_flux = interp1d(extended_wavelength, extended_flux, observed_wavelengths)
            print("Observed flux interpolated")

            # Final check for NaNs or infinities
            if torch.isnan(observed_flux).any() or torch.isinf(observed_flux).any():
                print("NaN or Inf detected in observed_flux")

            return observed_flux

        except Exception as e:
            print(f"Error in DownsamplingLayer forward: {e}")
            raise

class FullNetwork(nn.Module):
    def __init__(self, generator, high_res_wavelength, device):
        super(FullNetwork, self).__init__()
        self.generator = generator
        self.downsampling_layer = DownsamplingLayer()
        self.high_res_wavelength = torch.tensor(high_res_wavelength, dtype=torch.float16)
        self.device = device
    
    def forward(self, z, observed_wavelengths):
        print("Entering FullNetwork forward")
        try:
            high_res_flux = self.generator(z)
            print(f"Generator output - high_res_flux: shape={high_res_flux.shape}, dtype={high_res_flux.dtype}")
            
            if torch.isnan(high_res_flux).any() or torch.isinf(high_res_flux).any():
                print("NaN or Inf detected in generator output")

            downsampled_flux = self.downsampling_layer(high_res_flux, self.high_res_wavelength, observed_wavelengths, self.device)
            print("Downsampling completed")
            
            if torch.isnan(downsampled_flux).any() or torch.isinf(downsampled_flux).any():
                print("NaN or Inf detected in downsampled_flux")

            return downsampled_flux
        except Exception as e:
            print(f"Error in FullNetwork forward: {e}")
            raise
