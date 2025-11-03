""""
Helper functions to utilize optuna for hyperparameter optimization
"""
# Regressor head with flexible architecture for optimization

class OptimizedWav2Vec2Regression(nn.Module):
    def __init__(self, base_model, hidden_size, dropout_rate, num_layers=2):
        super().__init__()
        self.wav2vec2 = base_model

        # Build dynamic regressor based on trial parameters
        layers = []
        input_size = self.wav2vec2.config.hidden_size

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_size = hidden_size

        # Final output layer
        layers.append(nn.Linear(input_size, 1))

        self.regressor = nn.Sequential(*layers)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.regressor(pooled).squeeze(1)
    
class ComplexWav2Vec2Regression(nn.Module):
    def __init__(self, base_model, hidden_size, dropout_rate, num_layers=3, activation='gelu', use_batch_norm=True, pooling_method='mean', use_residual=True):
        super().__init__()
        self.wav2vec2 = base_model
        self.pooling_method = pooling_method
        self.use_residual = use_residual

        # Activation selection
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'swish': nn.SiLU()
        }
        act_fn = activations.get(activation, nn.ReLU())

        # Build regression layers
        layers = []
        input_size = self.wav2vec2.config.hidden_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        self.regressor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, 1)

        if use_residual:
            self.residual_projection = nn.Linear(self.wav2vec2.config.hidden_size, 1)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1) if self.pooling_method == 'mean' else outputs.last_hidden_state[:, 0, :]
        x = self.regressor(pooled)
        out = self.output_layer(x).squeeze(1)
        if self.use_residual:
            out += self.residual_projection(pooled).squeeze(1)
        return out
    
def create_optimized_model(base_model, config):
    """
    Factory function to create different Wav2Vec2 regression model variants.

    Args:
        base_model: Pretrained Wav2Vec2 model.
        config: Dictionary with keys:
            - model_type: 'simple', 'improved', or 'complex'
            - hidden_size: int
            - dropout_rate: float
            - num_layers: int
            - activation: str, optional (for 'complex')
            - use_batch_norm: bool, optional
            - pooling_method: str, optional ('mean' or 'first'), optional for 'complex'
            - use_residual: bool, optional (for improved versions)
    """
    model_type = config.get('model_type', 'simple')

    if model_type == 'simple':
        return OptimizedWav2Vec2Regression(
            base_model=base_model,
            hidden_size=config['hidden_size'],
            dropout_rate=config['dropout_rate'],
            num_layers=config.get('num_layers', 2)
        )
    elif model_type == 'improved':
        return ImprovedWav2Vec2Regression(
            base_model=base_model,
            hidden_size=config['hidden_size'],
            dropout_rate=config['dropout_rate'],
            num_layers=config.get('num_layers', 2),
            use_batch_norm=config.get('use_batch_norm', True)
        )
    elif model_type == 'complex':
        return ComplexWav2Vec2Regression(
            base_model=base_model,
            hidden_size=config['hidden_size'],
            dropout_rate=config['dropout_rate'],
            num_layers=config.get('num_layers', 3),
            activation=config.get('activation', 'gelu'),
            use_batch_norm=config.get('use_batch_norm', True),
            pooling_method=config.get('pooling_method', 'mean'),
            use_residual=config.get('use_residual', True)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def objective_valence(trial, train_dataset, val_dataset, test_dataset, target_name):
    """
    Optuna objective function optimized for R², with adaptive variance regularization
    and extended hyperparameter search.
    """
    print(f"\n🚀 Starting trial {trial.number} for {target_name}")

    # -------------------------
    # 🔧 Hyperparameter search space
    # -------------------------
    model_type = trial.suggest_categorical("model_type", ["simple", "improved"])
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
    pooling_method = trial.suggest_categorical("pooling_method", ["mean", "max", "attention"])
    use_residual = trial.suggest_categorical("use_residual", [True, False])
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])

    # -------------------------
    # 📦 Dataloaders
    # -------------------------
    train_loader, val_loader, _ = create_dataloaders_with_batch_size(
        train_dataset, val_dataset, test_dataset, batch_size, collate_fn
    )

    try:
        # -------------------------
        # 🧠 Model creation
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
        }

        model = create_optimized_model(base_model, model_config).to(device)

        # -------------------------
        # ⚙️ Optimizer and Scheduler
        # -------------------------
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
        elif scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            scheduler = None

        # -------------------------
        # 🎯 Criterion
        # -------------------------
        if criterion_name == "MSELoss":
            criterion = nn.MSELoss()
        elif criterion_name == "SmoothL1Loss":
            criterion = nn.SmoothL1Loss(beta=0.5)
        else:
            criterion = nn.L1Loss()

        # -------------------------
        # 📏 Target normalization
        # -------------------------
        target_stats = None
        if normalize_targets:
            all_targets = []
            for _, targets in train_loader:
                all_targets.append(targets.numpy())
            all_targets = np.concatenate(all_targets)
            std = np.std(all_targets)
            std = std if std > 1e-6 else 1.0
            target_stats = {'mean': np.mean(all_targets), 'std': std}

        # -------------------------
        # 🚀 Train model
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
            target_stats=target_stats,
            normalize_targets=normalize_targets,
            patience=patience,
            min_delta=min_delta,
            variance_reg_coeff=variance_reg_coeff,
            var_reg_target_ratio=var_reg_target_ratio,
            freeze_backbone_epochs=freeze_backbone_epochs,
            prioritize_r2=prioritize_r2,
            verbose=True
        )

        # -------------------------
        # 🏆 Evaluation Metric
        # -------------------------
        if prioritize_r2:
            objective_value = -best_val_r2  # maximize R² by minimizing its negative
            print(f"✅ Trial {trial.number} completed - Best Val R²: {best_val_r2:.4f}")
        else:
            best_val_loss = min(val_losses)
            objective_value = best_val_loss
            print(f"✅ Trial {trial.number} completed - Best Val Loss: {best_val_loss:.4f}")

        return objective_value

    except Exception as e:
        print(f"❌ Trial {trial.number} failed with error: {e}")
        return float('inf')

    finally:
        del model, optimizer, scheduler
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
    if target_type.lower() == "valence":
        objective_func = objective_valence
    elif target_type.lower() == "arousal":
      objective_func = objective_arousal
    else:
      raise ValueError(f"Unknown target type: {target_type}")

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