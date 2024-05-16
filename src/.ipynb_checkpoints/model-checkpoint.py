import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, layers, activation):
        super(Generator, self).__init__()
        modules = []
        input_dim = latent_dim
        for layer_dim in layers:
            modules.append(nn.Linear(input_dim, layer_dim))
            modules.append(activation())
            input_dim = layer_dim
        modules.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, z):
        return self.model(z)
