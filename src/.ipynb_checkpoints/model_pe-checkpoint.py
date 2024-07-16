import torch
import torch.nn as nn
from pe import RandGaus

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, layers, max_wavelength=17500, activation_function='LeakyReLU', pe_dim=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_wavelength = float(max_wavelength)

        # Initialize RandGaus
        pe_args = (output_dim, pe_dim, 1.0, 1.0, False, 0)
        self.wavelength_encoder = RandGaus(pe_args)

        modules = []
        input_dim = latent_dim + pe_dim
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

    def forward(self, z, wavelengths):
        # print("z shape:", z.shape)
        # print("wavelengths shape:", wavelengths.shape)
        
        batch_size = z.size(0)
        num_wavelengths = wavelengths.size(1)
    
        # Normalize wavelengths
        normalized_wavelengths = wavelengths / self.max_wavelength
    
        # Encode wavelengths
        wavelength_encodings = self.wavelength_encoder.positional_encoding(normalized_wavelengths)
        # print("wavelength_encodings shape:", wavelength_encodings.shape)
        
        # Repeat z for each wavelength and concatenate with encodings
        z_repeated = z.unsqueeze(1).repeat(1, num_wavelengths, 1)
        # print("z_repeated shape:", z_repeated.shape)
        
        input_embeds = torch.cat([z_repeated, wavelength_encodings], dim=-1)
        # print("input_embeds shape before reshape:", input_embeds.shape)
        
        # Reshape for the linear layers
        input_embeds = input_embeds.view(batch_size * num_wavelengths, -1)
        # print("input_embeds shape after reshape:", input_embeds.shape)
        
        # Generate the complete spectrum
        spectrum = self.model(input_embeds)
        
        # Reshape back to (batch_size, num_wavelengths, output_dim)
        spectrum = spectrum.view(batch_size, num_wavelengths, self.output_dim)
        
        return spectrum
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)