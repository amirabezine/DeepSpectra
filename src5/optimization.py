import torch.optim as optim

class OptimizerInitializer:
    def __init__(self, config, generator, latent_codes):
        self.config = config
        self.generator = generator
        self.latent_codes = latent_codes

    def initialize_optimizers(self):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.config.learning_rate)
        optimizer_l = optim.Adam([self.latent_codes], lr=self.config.latent_learning_rate)
        return optimizer_g, optimizer_l
