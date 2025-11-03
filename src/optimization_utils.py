""""
Helper functions to utilize optuna for hyperparameter optimization
"""
import torch
import torch.nn as nn
import optuna

from src.task_utils import collate_fn, split_data
from src.model_architecture import SingleRegression, FlexibleRegression
from transformers import Wav2Vec2Model
from src.train_test import train_with_validation

def create_optimized_model(base_model, config):
    """
    Factory function to create different regression model variants.
    Works with SingleRegression, FlexibleRegression, and ComplexRegression.
    """
    model_type = config.get("model_type", "flexible")
    num_outputs = config.get("num_outputs", 1)  # 🔸 toggleable output count

    if model_type == "single":
        return SingleRegression(
            base_model=base_model,
            hidden_size=config["hidden_size"],
            dropout_rate=config["dropout_rate"],
            num_layers=config.get("num_layers", 2),
            num_outputs=num_outputs  # allow flexibility
        )

    elif model_type == "flexible":
        return FlexibleRegression(
            base_model=base_model,
            hidden_size=config["hidden_size"],
            dropout_rate=config["dropout_rate"],
            num_layers=config.get("num_layers", 2),
            use_batch_norm=config.get("use_batch_norm", True),
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


def objective(trial, train_dataset, val_dataset, test_dataset, target_name):
    print(f"\nStarting trial {trial.number} for {target_name}")

    # -------------------------
    # Hyperparameter search space
    # -------------------------
    model_type = trial.suggest_categorical("model_type", ["single", "flexible", "complex"])
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
    # Dataloaders
    # -------------------------
    train_loader, val_loader, _ = create_dataloaders_with_batch_size(
        train_dataset, val_dataset, test_dataset, batch_size, collate_fn
    )

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
            "num_outputs": num_outputs,  # 🔸 toggleable output count
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
            prioritize_r2=prioritize_r2,
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


def setup_and_run_optimization(target_type, train_dataset, val_dataset, test_dataset, n_trials, timeout=3600):
    """
    Complete setup script for running optimization, then retraining and saving final model

    Args:
        target_type (str): Either "valence" or "arousal"
        train_dataset: Training dataset for the specified target
        val_dataset: Validation dataset for the specified target
        test_dataset: Test dataset for the specified target
        n_trials (int): Number of optimization trials to run
        timeout (int): Maximum time in seconds for optimization

    Returns:
        optuna.Study: The completed study object
    """
    import pickle

    # Loading model
    model_name = "facebook/wav2vec2-large-xlsr-53"

    print("Starting Optuna Optimization")
    print(f"Using model: {model_name}")
    print(f"Target: {target_type.upper()}")
    print("="*60)

    # Pick objective function
    objective_func = objective

    # Running optimization
    print(f"OPTIMIZING {target_type.upper()} MODEL")
    print("="*50)

    study = optuna.create_study(
          direction="minimize",
          pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
          sampler=optuna.samplers.TPESampler(seed=42),
          study_name=f"{target_type}_optuna_study"
      )

    # Create objective function with datasets
    objective_with_data = lambda trial: objective_func(
        trial,
        train_dataset,
        val_dataset,
        test_dataset,
        target_name=target_type
    )

    # Run optimization
    study.optimize(objective_with_data, n_trials=n_trials, timeout=timeout)

    print(f"\n✅ {target_type.title()} Optimization Completed!")

    # Save study in pickle form
    with open(f'{target_type}_optuna_study.pkl', 'wb') as f:
        pickle.dump(study, f)

    # Retrieve best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    print(f"\n✅ Best trial: {best_trial.number}")
    print(f"Best validation loss: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # --- Retraining final model with best hyperparameters ---
    print("\n Retraining final model with best hyperparameters...")

    # Dataloaders
    train_loader, val_loader, _ = create_dataloaders_with_batch_size(
        train_dataset, val_dataset, test_dataset, best_params['batch_size'], collate_fn
    )

    # Model
    base_model = Wav2Vec2Model.from_pretrained(model_name)
    model = create_optimized_model(base_model, {
        "model_type": best_params.get("model_type", "improved"),
        "hidden_size": best_params["hidden_size"],
        "dropout_rate": best_params["dropout_rate"],
        "num_layers": best_params["num_layers"],
        "use_batch_norm": best_params.get("use_batch_norm", True),
        "pooling_method": best_params.get("pooling_method", "mean"),
        "use_residual": best_params.get("use_residual", False),
        "activation": best_params.get("activation", "relu"),
    }).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )

    # Scheduler
    scheduler = None
    if best_params.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif best_params.get("scheduler") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    elif best_params.get("scheduler") == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif best_params.get("scheduler") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Criterion
    criterion_name = best_params.get("criterion", "MSELoss")
    if criterion_name == "MSELoss":
        criterion = nn.MSELoss()
    elif criterion_name == "SmoothL1Loss":
        criterion = nn.SmoothL1Loss()
    elif criterion_name == "L1Loss":
        criterion = nn.L1Loss()

    # Optional target normalization
    normalize_targets = best_params.get("normalize_targets", False)
    target_stats = None
    if normalize_targets:
        all_targets = []
        for _, targets in train_loader:
            all_targets.append(targets.numpy())
        all_targets = np.concatenate(all_targets)
        target_stats = {
            'mean': np.mean(all_targets),
            'std': np.std(all_targets)
        }

    # Get variance regularization coefficients
    variance_reg_coeff = best_params.get("variance_reg_coeff", 0.05)
    var_reg_target_ratio = best_params.get("var_reg_target_ratio", 0.5)


    train_losses, val_losses, best_val_r2 = train_with_validation(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        feature_extractor=feature_extractor,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=25,  # Slightly more epochs for final training
        sampling_rate=model_sr,
        grad_clip=best_params.get("grad_clip", 1.5),
        target_stats=target_stats,
        normalize_targets=normalize_targets,
        patience=best_params.get("patience", 5),
        min_delta=best_params.get("min_delta", 0.001),
        variance_reg_coeff=variance_reg_coeff,
        var_reg_target_ratio=var_reg_target_ratio,
        freeze_backbone_epochs=best_params.get("freeze_backbone_epochs", 2),
        prioritize_r2=best_params.get("prioritize_r2", True),
        verbose=True
    )

    trial_num = best_trial.number

    # Save final best model
    save_path = f"optimized_{target_type}_model_v4_{trial_num}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Final model saved at {save_path}")

    return study, save_path


import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader

# ✅ assume these are defined somewhere:
# - collate_fn (toggleable version)
# - split_data()
# - SingleLabelRegression, FlexibleRegression
# - Wav2Vec2Model, feature_extractor, train_with_validation, device, model_sr

def objective(trial, waveforms, targets, target_name):
    print(f"\n🚀 Starting trial {trial.number} for {target_name}")

    # -------------------------
    # Hyperparameter search space
    # -------------------------
    model_type = trial.suggest_categorical("model_type", ["single", "flexible", "complex"])
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
    # Pick dataset class dynamically
    # -------------------------
    dataset_class = SingleLabelRegression if num_outputs == 1 else FlexibleRegression

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
            "num_outputs": num_outputs,  # 🔸 toggleable output count
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
            prioritize_r2=prioritize_r2,
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
