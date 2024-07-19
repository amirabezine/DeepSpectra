from utils import get_config2, resolve_path

class Config:
    def __init__(self):
        self.config = get_config2()

    @property
    def dataset_name(self):
        return self.config['dataset_name']

    @property
    def data_path(self):
        return resolve_path(self.config['datasets'][self.dataset_name]['path'])

    @property
    def checkpoints_path(self):
        return resolve_path(self.config['paths']['checkpoints'])

    @property
    def latent_path(self):
        return resolve_path(self.config['paths']['latent'])

    @property
    def plots_path(self):
        return resolve_path(self.config['paths']['plots'])

    @property
    def batch_size(self):
        return self.config['training']['batch_size']

    @property
    def num_workers(self):
        return self.config['training']['num_workers']

    @property
    def num_epochs(self):
        return self.config['training']['num_epochs']

    @property
    def learning_rate(self):
        return self.config['training']['learning_rate']

    @property
    def latent_learning_rate(self):
        return self.config['training']['latent_learning_rate']

    @property
    def latent_dim(self):
        return self.config['training']['latent_dim']

    @property
    def checkpoint_interval(self):
        return self.config['training']['checkpoint_interval']

    @property
    def max_files(self):
        return self.config['training']['max_files']

    @property
    def n_subspectra(self):
        return self.config['training']['n_subspectra']

    @property
    def split_ratios(self):
        return self.config['training']['split_ratios']

    @property
    def scheduler_gamma(self):
        return self.config['training']['scheduler_gamma']

    @property
    def scheduler_step_size(self):
        return self.config['training']['scheduler_step_size']

    @property
    def n_samples_per_spectrum(self):
        return self.config['training']['n_samples_per_spectrum']

    @property
    def validation_split(self):
        return self.config['training']['validation_split']

    @property
    def generator_layers(self):
        return self.config['model']['generator_layers']

    @property
    def activation_function(self):
        return self.config['model']['activation_function']

    @property
    def output_dim(self):
        return self.config['model']['output_dim']

    @property
    def max_wavelength(self):
        return self.config['model']['max_wavelength']

    @property
    def pe_dim(self):
        return self.config['model']['pe_dim']
