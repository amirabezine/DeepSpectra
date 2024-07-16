import torch
import torch.nn as nn
import numpy as np

class RandGaus(nn.Module):
    def __init__(self, pe_args):
        super(RandGaus, self).__init__()
        (dim, pe_dim, omega, sigma, pe_bias, seed) = pe_args
        self.dim = dim
        self.omega = omega
        self.sigma = sigma
        self.bias = pe_bias
        self.pe_dim = pe_dim
        
        torch.manual_seed(seed)
        self.mapping = nn.Linear(1, self.pe_dim, bias=self.bias)

    def positional_encoding(self, wavelengths):
        """
        Method to add positional encoding to the given wavelengths
        """
        wavelengths = wavelengths.unsqueeze(-1)  # Add dimension for linear layer
        
        pe = torch.sin(self.omega * self.mapping(wavelengths))
        
        return pe