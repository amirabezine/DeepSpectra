import torch
import torch.nn as nn
import torch.nn.functional as F
from pe import PositionalEncoding


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DownsamplingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DownsamplingLayer, self).__init__()
        self.downsample = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.downsample(x)
        return x

class FullNetwork(nn.Module):
    def __init__(self, latent_dim, pe_args, hidden_dim, wavelength_grid_dim, real_wavelength_dim):
        super(FullNetwork, self).__init__()
        self.positional_encoding = PositionalEncoding(pe_args)
        self.generator = Generator(latent_dim + pe_args[1], hidden_dim, wavelength_grid_dim)
        self.downsampling_layer = DownsamplingLayer(wavelength_grid_dim, real_wavelength_dim)
        self.latent_dim = latent_dim
        self.real_wavelength_dim = real_wavelength_dim

    def forward(self, latent, wavelength_grid):
        # Generate positional encoding
        pe = self.positional_encoding(wavelength_grid)
        
        # Expand latent vector to match the batch size and concatenate with positional encoding
        latent_expanded = latent.unsqueeze(1).expand(-1, wavelength_grid.size(1), -1)
        x = torch.cat((latent_expanded, pe), dim=-1)
        
        # Generate high-resolution intermediate spectrum
        high_res_spectrum = self.generator(x)
        
        # Downsample to match observed spectrum resolution
        low_res_spectrum = self.downsampling_layer(high_res_spectrum)
        
        return low_res_spectrum