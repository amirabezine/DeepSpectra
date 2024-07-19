import os
import yaml
import csv
import torch

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_config():
    config_path = os.path.join(get_project_root(), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config2():
    config_path = os.path.join(get_project_root(), 'config', 'config2.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def resolve_path(path):
    return os.path.join(get_project_root(), path)

def generate_latent_codes(max_files, latent_size):
    """Generate random latent codes for each shape from a Gaussian distribution.

    Returns:
        - latent_codes: np.array, shape (num_shapes, latent_size)
        - dict_latent_codes: key: obj_index, value: corresponding idx in the latent_codes array. 
                             e.g., latent_codes = ([ [1, 2, 3], [7, 8, 9] ])
                             dict_latent_codes[345] = 0, the obj that has index 345 refers to 
                             the 0-th latent code.
    """
    latent_codes = torch.tensor([], dtype=torch.float32).reshape(0, latent_size).to(device)
    for i, obj_idx in enumerate(list(samples_dict.keys())):
        latent_code = torch.normal(0, 0.01, size=(1, latent_size), dtype=torch.float32).to(device)
        latent_codes = torch.vstack((latent_codes, latent_code))
    latent_codes.requires_grad_(True)
    return latent_codes

def save_timing_data(filepath, timing_data):
    """Save timing data to a CSV file.

    Args:
        filepath (str): The file path to save the timing data.
        timing_data (list): A list of tuples containing timing data.
    """
    file_exists = os.path.isfile(filepath)
    try:
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Epoch', 'Total Training Time', 'Total Validation Time', 'Weight Optimization Time', 'Latent Optimization Time'])
            for data in timing_data:
                writer.writerow(data)
    except OSError as e:
        print(f"Failed to open file {filepath}: {e}")
