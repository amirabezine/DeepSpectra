import torch
import torch.nn as nn
from torchinterp1d import interp1d
from torch.autograd import Function

class InterpolateFunction(Function):
    @staticmethod
    def forward(ctx, high_res_wavelength, high_res_spectra, observed_wavelengths, intervals):
        # Save tensors for the backward pass
        ctx.save_for_backward(high_res_wavelength, high_res_spectra, observed_wavelengths)
        ctx.intervals = intervals

        device = high_res_wavelength.device  # Ensure all operations are on the same device
        batch_size, _ = observed_wavelengths.shape
        interpolated_spectrum = torch.zeros_like(observed_wavelengths)

        for i in range(batch_size):
            high_res_spectrum = high_res_spectra[i].to(device)  # Move to the correct device
            for start, end in intervals:
                mask_hr = (high_res_wavelength >= start) & (high_res_wavelength <= end)
                if torch.any(mask_hr):
                    interval_wavelengths = high_res_wavelength[mask_hr]
                    interval_spectrum = high_res_spectrum[mask_hr]

                    # Padding and interpolation as previously described
                    padding_amount = 10
                    padded_wavelengths = torch.cat([
                        torch.linspace(start - padding_amount, start, padding_amount // 2, device=device),
                        interval_wavelengths,
                        torch.linspace(end, end + padding_amount, padding_amount // 2, device=device)
                    ])

                    padded_spectrum = torch.cat([
                        interval_spectrum[:1].repeat(padding_amount // 2),
                        interval_spectrum,
                        interval_spectrum[-1:].repeat(padding_amount // 2)
                    ])

                    mask_obs = (observed_wavelengths[i] >= start - padding_amount) & (observed_wavelengths[i] <= end + padding_amount)
                    valid_obs_wavelengths = observed_wavelengths[i][mask_obs].to(device)

                    if len(valid_obs_wavelengths) > 0:
                        interp_values = interp1d(padded_wavelengths.to(device), padded_spectrum.to(device), valid_obs_wavelengths)
                        interpolated_spectrum[i][mask_obs] = interp_values

        return interpolated_spectrum

    @staticmethod
    def backward(ctx, grad_output):
        high_res_wavelength, high_res_spectra, observed_wavelengths = ctx.saved_tensors
        intervals = ctx.intervals

        grad_high_res_wavelength = None  # No gradient necessary for wavelengths
        grad_high_res_spectra = torch.zeros_like(high_res_spectra)
        grad_observed_wavelengths = None

        return grad_high_res_wavelength, grad_high_res_spectra, grad_observed_wavelengths, None

class SpectrumGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, layers, activation_function='LeakyReLU'):
        super(SpectrumGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.generator = Generator(latent_dim, output_dim, layers, activation_function)
        self.high_res_wavelength = self.generate_wavelength_grid()
        
        self.intervals = [
            (4711, 4906),
            (5647, 5875),
            (6475, 6737),
            (7583, 7885),
            (15100, 17000)
        ]
        

    def forward(self, z, observed_wavelengths):
        high_res_spectrum = self.generator(z)
        interpolated_spectrum = InterpolateFunction.apply(self.high_res_wavelength, high_res_spectrum, observed_wavelengths, self.intervals)
        return interpolated_spectrum

    def generate_wavelength_grid(self):
        grid = [
            (15050, 15850, 0.2),
            (15870, 16440, 0.2),
            (16475, 17100, 0.2),
            (4700, 4930, 0.05),
            (5630, 5880, 0.05),
            (6420, 6800, 0.05),
            (7500, 7920, 0.05)
        ]
        wavelength_grid = []
        for start, end, step in grid:
            wavelength_grid.extend(torch.arange(start, end + step, step))
        return torch.tensor(wavelength_grid, dtype=torch.float32)


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
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.model(z)
