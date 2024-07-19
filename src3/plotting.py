import os
import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, config):
        self.config = config

    def plot_losses(self):
        loss_history_path = os.path.join(self.config.checkpoints_path, 'loss_history.npy')
        plots_path = self.config.plots_path
        loss_history = np.load(loss_history_path, allow_pickle=True).item()
        train_losses = loss_history['train']
        val_losses = loss_history['val']
        before_losses = loss_history['before']
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.plot(before_losses, label='Training Loss before latent optimization')
        plt.title('Training and Validation Losses Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0, 10)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_path, f'lossesandbeforeopt.png'))
