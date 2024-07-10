import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, layers, activation_function='LeakyReLU'):
        super(Generator, self).__init__()
        modules = []
        input_dim = latent_dim
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
        return self.model(z)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def generate_spectrum(self, z, wavelengths):
        z_expanded = z.unsqueeze(1).expand(-1, wavelengths.size(0), -1)
        wavelengths_expanded = wavelengths.unsqueeze(0).expand(z.size(0), -1).unsqueeze(-1)
        input_tensor = torch.cat([z_expanded, wavelengths_expanded], dim=-1)
        return self.model(input_tensor.view(-1, self.latent_dim + 1)).view(z.size(0), -1)
