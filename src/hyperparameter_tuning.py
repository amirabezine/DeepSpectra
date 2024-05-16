import optuna
import yaml
import torch
from train import train_glo

def objective(trial):
    # Load the base config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Suggest hyperparameters
    config['training']['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    config['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    config['model']['latent_dim'] = trial.suggest_int('latent_dim', 20, 100)
    config['training']['num_epochs'] = trial.suggest_int('num_epochs', 50, 200)
    
    # Train the model
    _, loss_history = train_glo(config)
    
    # Return the best validation loss
    return min(loss_history)

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters: ", study.best_params)
