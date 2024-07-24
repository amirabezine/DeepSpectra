import torch
import torch.nn as nn
from torchinterp1d import interp1d
from torch.autograd import Function


class InterpolateFunction(Function):
    @staticmethod
    def forward(ctx, high_res_wavelength, high_res_spectrum, observed_wavelengths):
        ctx.save_for_backward(high_res_wavelength, high_res_spectrum, observed_wavelengths)
        
        # Correctly pass all three arguments to interp1d
        interpolated_spectrum = interp1d(
            high_res_wavelength,
            high_res_spectrum,
            observed_wavelengths
        )
        
        return interpolated_spectrum

    @staticmethod
    def backward(ctx, grad_output):
        high_res_wavelength, high_res_spectrum, observed_wavelengths = ctx.saved_tensors
        # Initialize gradients for all inputs
        grad_high_res_wavelength = grad_high_res_spectrum = grad_observed_wavelengths = None
        
        # If you need to compute gradients for these, replace the None values with actual gradient computations
        # Example:
        # grad_high_res_spectrum = some_gradient_computation_function(grad_output, ...)
        
        return grad_high_res_wavelength, grad_high_res_spectrum, grad_observed_wavelengths



class SpectrumGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, layers, activation_function='LeakyReLU'):
        super(SpectrumGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Initialize the generator
        self.generator = Generator(latent_dim, output_dim, layers, activation_function)
        
        # Generate high-resolution wavelength grid
        self.high_res_wavelength = self.generate_wavelength_grid()
        print(f"High-res wavelength shape: {self.high_res_wavelength.shape}")

    def forward(self, z, observed_wavelengths):
        # Generate high-resolution spectrum
        high_res_spectrum = self.generator(z)
        
        # Apply custom interpolation function
        interpolated_spectrum = InterpolateFunction.apply(self.high_res_wavelength, high_res_spectrum, observed_wavelengths)
        
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


    def downsample_spectrum(self, high_res_spectrum, observed_wavelengths):
        # print("Starting spectrum downsampling.")
        batch_size = high_res_spectrum.size(0)
        
        # Ensure all tensors are on the same device
        high_res_wavelength = self.high_res_wavelength.to(high_res_spectrum.device)
        observed_wavelengths = observed_wavelengths.to(high_res_spectrum.device)
        
        # Interpolation
        interpolated_spectrum = interp1d(
            high_res_wavelength.unsqueeze(0).expand(batch_size, -1),
            high_res_spectrum,
            observed_wavelengths
        )
        # print("Downsampling completed.")
        
        return interpolated_spectrum

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, layers, activation_function='LeakyReLU'):
        super(Generator, self).__init__()
        print("Initializing Generator.")
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
        print("Generator initialized with final output dimension:", output_dim)

    def forward(self, z):
        # print("Entering forward of Generator.")
        output = self.model(z)
        # print("Generator forward pass completed with output shape:", output.shape)
        return output

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):  # Check if the module is an instance of nn.Linear
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            print(f"Initialized weights for {m.__class__.__name__} with shape: {m.weight.shape}")
        else:
            print(f"No weights to initialize for {m.__class__.__name__}")