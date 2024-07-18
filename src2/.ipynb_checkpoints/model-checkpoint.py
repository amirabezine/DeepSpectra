import torch
import torch.nn as nn
import torch.nn.functional as F
from pe import PositionalEncoding

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, layers, activation_function='LeakyReLU'):
        super(Generator, self).__init__()
        modules = []
        print("output_dim  generr = " , output_dim)
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

    def forward(self, z):
        batch_size, num_wavelengths, input_dim = z.shape
        # Flatten the input to apply the model
        z = z.view(-1, input_dim)
        output = self.model(z)
        # Reshape the output back to the original dimensions
        output = output.view(batch_size, num_wavelengths, -1)
        return output


class DownsamplingLayer(nn.Module):
    def __init__(self):
        super(DownsamplingLayer, self).__init__()

    def forward(self, high_res_flux, high_res_wavelength, observed_wavelength):
        print(f"high_res_flux shape: {high_res_flux.shape}")
        print(f"high_res_wavelength shape: {high_res_wavelength.shape}")
        print(f"observed_wavelength shape: {observed_wavelength.shape}")

        # Add channel and spatial dimensions to high_res_flux
        high_res_flux = high_res_flux.unsqueeze(1).unsqueeze(-1)  # [batch_dim, 1, high_res_dim, 1]
        print(f"Reshaped high_res_flux shape: {high_res_flux.shape}")

        # Normalize observed_wavelength to be in the range [-1, 1]
        grid = (observed_wavelength - high_res_wavelength.min()) / (high_res_wavelength.max() - high_res_wavelength.min()) * 2 - 1
        
        # Reshape grid for 2D sampling
        grid = grid.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, 2)  # [batch_dim, 1, real_wavelength_dim, 2]
        print(f"Grid shape: {grid.shape}")

        # Interpolate
        sampled_flux = F.grid_sample(high_res_flux, grid, align_corners=True, mode='bilinear', padding_mode='border')
        print(f"Sampled flux shape: {sampled_flux.shape}")
        
        return sampled_flux.squeeze(1).squeeze(-1)  # Remove extra dimensions


class FullNetwork(nn.Module):
    def __init__(self, generator, downsampling_layer, positional_encoding):
        super(FullNetwork, self).__init__()
        self.generator = generator
        self.downsampling_layer = downsampling_layer
        self.positional_encoding = positional_encoding

    def forward(self, latent_z, wavelength_grid, real_wavelengths):
        print(f"latent_z shape: {latent_z.shape}")
        print(f"wavelength_grid shape: {wavelength_grid.shape}")
        print(f"real_wavelengths shape: {real_wavelengths.shape}")

        batch_dim = latent_z.size(0)
        
        # Generate positional encoding
        wavelength_grid_expanded = wavelength_grid.unsqueeze(0).expand(batch_dim, -1)
        positional_encoding = self.positional_encoding(wavelength_grid_expanded)
        print(f"positional_encoding shape: {positional_encoding.shape}")

        # Expand latent_z to match the shape of positional_encoding
        latent_z_expanded = latent_z.unsqueeze(1).expand(-1, positional_encoding.size(1), -1)
        print(f"latent_z_expanded shape: {latent_z_expanded.shape}")

        # Concatenate latent_z and positional_encoding
        input_to_generator = torch.cat((latent_z_expanded, positional_encoding), dim=-1)
        print(f"input_to_generator shape: {input_to_generator.shape}")

        # Generate high-resolution intermediate spectrum
        # print(input_to_generator.view(-1, input_to_generator.size(-1)).shape)
        
        # generator_output = self.generator(input_to_generator.view(-1, input_to_generator.size(-1)))

        generator_output = self.generator(input_to_generator)
        print(f"generator_output shape: {generator_output.shape}")
        generator_output_seq = generator_output.squeeze(-1)
        print("generator_output squeezed shape: ", generator_output_seq.shape)
        high_res_flux = generator_output.view(batch_dim, wavelength_grid.size(0))
        print(f"high_res_flux shape: {high_res_flux.shape}")

        # Interpolate to real wavelengths
        generated_flux = self.downsampling_layer(high_res_flux, wavelength_grid, real_wavelengths)
        print(f"generated_flux shape: {generated_flux.shape}")
        
        return generated_flux
