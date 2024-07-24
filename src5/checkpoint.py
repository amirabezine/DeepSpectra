import os
import torch

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        print(f"Checkpoint loaded: {filename}")
        return checkpoint
    else:
        print(f"No checkpoint found at: {filename}")
        return None

class CheckpointManager:
    def __init__(self, config, generator, optimizer_g, optimizer_l, device, train_loader):
        self.config = config
        self.generator = generator
        self.optimizer_g = optimizer_g
        self.optimizer_l = optimizer_l
        self.device = device
        self.train_loader = train_loader

    def save_checkpoint(self, state, filename):
        save_checkpoint(state, filename)

    def load_checkpoints(self):
        checkpoints_path = self.config.checkpoints_path
        latest_checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_latest.pth.tar')
        best_checkpoint_path = os.path.join(checkpoints_path, 'checkpoint_best.pth.tar')
        latest_checkpoint = load_checkpoint(latest_checkpoint_path)
        
        start_epoch = 0
        best_val_loss = float('inf')
        latent_codes = None
        dict_latent_codes = None

        if latest_checkpoint:
            try:
                self.generator.load_state_dict(latest_checkpoint['state_dict'])
                self.optimizer_g.load_state_dict(latest_checkpoint['optimizer_g_state'])
                self.optimizer_l.load_state_dict(latest_checkpoint['optimizer_l_state'])
                latent_codes = latest_checkpoint['latent_codes'].to(self.device)
                dict_latent_codes = latest_checkpoint['dict_latent_codes']
                start_epoch = latest_checkpoint['epoch']
                best_val_loss = latest_checkpoint['best_loss']
            
                print(f"Loaded latest checkpoint. Starting from epoch {start_epoch}")
            except KeyError as e:
                print(f"Error loading state dictionaries from latest checkpoint: {e}")
                print("Initializing from scratch.")
        
        if latent_codes is None or dict_latent_codes is None:
            print("Loading latent codes from dataset...")
            latent_codes, dict_latent_codes = self.train_loader.dataset.load_latent_vectors([self.train_loader], self.config.latent_dim, self.device)
            
            # Recreate optimizer_l with the loaded latent_codes
            self.optimizer_l = torch.optim.Adam([latent_codes], lr=self.config.latent_learning_rate)

        scheduler_g = torch.optim.lr_scheduler.StepLR(self.optimizer_g, step_size=self.config.scheduler_step_size, gamma=self.config.scheduler_gamma)
        scheduler_l = torch.optim.lr_scheduler.StepLR(self.optimizer_l, step_size=self.config.scheduler_step_size, gamma=self.config.scheduler_gamma)

        # Load best validation loss if not found in latest checkpoint
        if best_val_loss == float('inf'):
            best_checkpoint = load_checkpoint(best_checkpoint_path)
            if best_checkpoint:
                try:
                    best_val_loss = best_checkpoint['best_loss']
                    print(f"Best validation loss from checkpoint: {best_val_loss}")
                except KeyError as e:
                    print(f"Error retrieving best validation loss from checkpoint: {e}")

        return start_epoch, scheduler_g, scheduler_l, best_val_loss, latent_codes, dict_latent_codes, self.optimizer_l
