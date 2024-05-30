import torch
import os

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Save a training checkpoint.
    
    Args:
        state (dict): Contains the model's state, optimizer's state, and other variables.
        filename (str): Path where the checkpoint is saved.
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path):
    """Load a training checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file to be loaded.
    
    Returns:
        dict: The loaded checkpoint.
    """
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return None
