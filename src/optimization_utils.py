""""
Helper functions to utilize optuna for hyperparameter optimization
"""
import torch
import torch.nn as nn
import optuna

from src.task_utils import collate_fn, split_data, SingleRegression, FlexibleRegression
from src.models import OptimizedRegression, ComplexRegression
from transformers import Wav2Vec2Model
from src.train_test import train_with_validation

def create_optimized_model(base_model, config):
    """
    Factory function to create different regression model variants.
    Works with OptimizedRegression and ComplexRegression
    """
    model_type = config.get("model_type", "flexible")
    num_outputs = config.get("num_outputs", 1)  # toggleable output count

    if model_type == "simple":
        return OptimizedRegression(
            base_model=base_model,
            hidden_size=config["hidden_size"],
            dropout_rate=config["dropout_rate"],
            num_layers=config.get("num_layers", 2),
            num_outputs=num_outputs
        )

    elif model_type == "complex":
        return ComplexRegression(
            base_model=base_model,
            hidden_size=config["hidden_size"],
            dropout_rate=config["dropout_rate"],
            num_layers=config.get("num_layers", 3),
            activation=config.get("activation", "gelu"),
            use_batch_norm=config.get("use_batch_norm", True),
            pooling_method=config.get("pooling_method", "mean"),
            use_residual=config.get("use_residual", True),
            num_outputs=num_outputs
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def objective(trial, waveforms, targets, target_name, model_name, model_sr, feature_extractor, device):
    print(f"\n Starting trial {trial.number} for {target_name}")
    print(f"Trial {trial.number} params: {trial.params}")

    # -------------------------
    # Hyperparameter search space
    # -------------------------
    model_type = trial.suggest_categorical("model_type", ["simple", "complex"])
    num_outputs = trial.suggest_int("num_outputs", 1, 2)  # allows single vs double regression

    learning_rate = trial.suggest_float("learning_rate", 1.5e-5, 3.5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 768])
    dropout_rate = trial.suggest_float("dropout_rate", 0.15, 0.35)
    num_layers = trial.suggest_int("num_layers", 2, 4)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 3e-4, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    scheduler_type = trial.suggest_categorical("scheduler", ["step", "cosine", "exponential"])
    criterion_name = trial.suggest_categorical("criterion", ["MSELoss", "SmoothL1Loss"])
    normalize_targets = trial.suggest_categorical("normalize_targets", [False, True])
    grad_clip = trial.suggest_float("grad_clip", 1.0, 2.0)
    variance_reg_coeff = trial.suggest_float("variance_reg_coeff", 0.05, 0.15)
    var_reg_target_ratio = trial.suggest_float("var_reg_target_ratio", 0.4, 0.7)
    patience = trial.suggest_int("patience", 3, 8)
    prioritize_r2 = trial.suggest_categorical("prioritize_r2", [True, False])
    freeze_backbone_epochs = trial.suggest_int("freeze_backbone_epochs", 1, 4)
    min_delta = trial.suggest_float("min_delta", 1e-4, 1e-2, log=True)
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    pooling_method = trial.suggest_categorical("pooling_method", ["mean", "first"])
    use_residual = trial.suggest_categorical("use_residual", [True, False])
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])

    # -------------------------
    # Pick dataset class based off model type
    # -------------------------
    if num_outputs == 1:
        dataset_class = SingleRegression
    else:
        dataset_class = FlexibleRegression
    # -------------------------
    # Create data loaders using split_data()
    # -------------------------
    loaders = split_data(
        waveforms=waveforms,
        targets=targets,
        target_name=target_name,
        batch_size=batch_size,
        collate_fn=collate_fn,
        dataset_class=dataset_class
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    try:
        # -------------------------
        # Model creation
        # -------------------------
        base_model = Wav2Vec2Model.from_pretrained(model_name)

        model_config = {
            "model_type": model_type,
            "hidden_size": hidden_size,
            "dropout_rate": dropout_rate,
            "num_layers": num_layers,
            "use_batch_norm": use_batch_norm,
            "pooling_method": pooling_method,
            "use_residual": use_residual,
            "activation": activation,
            "num_outputs": num_outputs,  # toggleable output count
        }

        model = create_optimized_model(base_model, model_config).to(device)

        # -------------------------
        # Optimizer, Scheduler, Loss
        # -------------------------
        optimizer = (
            torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            if optimizer_name == "Adam"
            else torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        )

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
        elif scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            scheduler = None

        criterion = nn.MSELoss() if criterion_name == "MSELoss" else nn.SmoothL1Loss(beta=0.5)

        # -------------------------
        # Train
        # -------------------------
        train_losses, val_losses, best_val_r2 = train_with_validation(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            feature_extractor=feature_extractor,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            num_epochs=30,
            sampling_rate=model_sr,
            trial=trial,
            grad_clip=grad_clip,
            normalize_targets=normalize_targets,
            patience=patience,
            min_delta=min_delta,
            variance_reg_coeff=variance_reg_coeff,
            var_reg_target_ratio=var_reg_target_ratio,
            freeze_backbone_epochs=freeze_backbone_epochs,
            prioritize_r2=True,
            verbose=True,
        )

        # -------------------------
        # Objective
        # -------------------------
        return -best_val_r2 if prioritize_r2 else min(val_losses)

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float("inf")

    finally:
        torch.cuda.empty_cache()


def setup_and_run_optimization(target_type, waveforms, targets, n_trials, timeout, model_name, model_sr, feature_extractor, device):
    import pickle

    print("Starting Optuna Optimization")
    print(f"Target: {target_type.upper()}")
    print("=" * 60)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"{target_type}_optuna_study"
    )

    objective_with_data = lambda trial: objective(
        trial,
        waveforms=waveforms,
        targets=targets,
        target_name=target_type,
        model_name=model_name,
        model_sr=model_sr,
        feature_extractor=feature_extractor,
        device=device,
    )

    # Run optimization
    study.optimize(objective_with_data, n_trials=n_trials, timeout=timeout)

    print(f"\n✅ {target_type.title()} Optimization Completed!")

    # Save study in pickle form
    with open(f'{target_type}_optuna_study.pkl', 'wb') as f:
        pickle.dump(study, f)

    # Print best results
    best_trial = study.best_trial
    print(f"\n Best trial: {best_trial.number}")

    if best_trial.params.get("prioritize_r2", True):
        print(f"Best validation R²: {-best_trial.value:.4f}")
    else:
        print(f"Best validation loss: {best_trial.value:.4f}")

    print("Best hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    return study
