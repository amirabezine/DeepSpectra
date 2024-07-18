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
        self.pe_dim = pe_dim // 2

        self.mappings = nn.ModuleList([self.init_mapping(i + seed) for i in range(dim)])

    def init_mapping(self, seed):
        torch.manual_seed(seed)  # Set seed for reproducibility
        mapping = nn.Linear(1, self.pe_dim, bias=self.bias)
        mapping.weight = nn.Parameter(self.randmz_weights(seed).to(torch.float32), requires_grad=False)
        if self.bias:
            mapping.bias = nn.Parameter(mapping.bias.to(torch.float32), requires_grad=False)
        return mapping

    def randmz_weights(self, seed):
        torch.manual_seed(seed)  # Set seed for reproducibility
        weight = torch.empty((self.pe_dim, 1), dtype=torch.float32).normal_(mean=0., std=self.sigma**2)
        weight = 2 * torch.pi * self.omega * weight
        return weight

    def forward(self, coords):
        coords = coords.to(torch.float32)  # Use float32 for intermediate computations
        encd_coords = self.mappings[0](coords[..., 0:1])
        for i in range(1, self.dim):
            encd_coords += self.mappings[i](coords[..., i:i+1])
        result = torch.cat((torch.cos(encd_coords), torch.sin(encd_coords)), dim=-1)
        return result.to(torch.float16)  # Cast back to float16 for the final result

class PositionalEncoding(nn.Module):
    def __init__(self, pe_args):
        super(PositionalEncoding, self).__init__()
        self.rand_gaus = RandGaus(pe_args)

    def forward(self, wavelength_grid):
        # Expand dimensions to match expected input format: [..., bsz, (nsmpl,) dim]
        wavelength_grid = wavelength_grid.unsqueeze(-1)  # [..., dim]
        pe = self.rand_gaus(wavelength_grid)
        return pe