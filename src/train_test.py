import tqdm
import torch
import numpy as np
import math
import optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt

def preprocess_batch(waveforms, extractor, sampling_rate=16000):
    """
    Preprocess batch of waveforms for Wav2Vec2 model
    Args:
        waveforms: Tensor [B, T] - batch of audio waveforms
        sampling_rate: int - sampling rate (16000 Hz for Wav2Vec2)
    Returns:
        input_values: preprocessed audio tensors
        attention_mask: attention mask for padding
    """
    # waveforms: Tensor [B, T]
    waveform_list = [waveforms[i].cpu().numpy() for i in range(waveforms.shape[0])]

    # Using feature extractor for the multilingual model
    inputs = extractor(waveform_list, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    return inputs.input_values, inputs.attention_mask

def train_with_validation(
    model,
    train_dataloader,
    val_dataloader,
    feature_extractor,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs,
    sampling_rate,
    trial=None,
    grad_clip=1.5,
    target_stats=None,
    normalize_targets=False,
    patience=5,         # early stopping patience
    min_delta=1e-3,     # min improvement to be considered for early stopping
    variance_reg_coeff=0.05,  # weight for variance regularization
    freeze_backbone_epochs=3,
    verbose=True
):
    """
    Training function with mode collapse fixes
    """
    model.to(device)
    train_losses = []
    val_losses = []

    # Var initialization for early stopping based off val loss
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Var initialization for early stopping based off r2
    best_val_r2 = -float("inf")
    epochs_no_improve_r2 = 0


    for epoch in range(num_epochs):
       # Optional freezing backbone
        if freeze_backbone_epochs > 0 and epoch < freeze_backbone_epochs:
            for param in model.wav2vec2.parameters():
                param.requires_grad = False
        else:
            for param in model.wav2vec2.parameters():
                param.requires_grad = True

        # Training phase
        model.train()
        running_train_loss = 0.0
        running_output_sum = 0.0
        running_output_sq_sum = 0.0
        running_target_sum = 0.0
        running_target_sq_sum = 0.0
        total_samples = 0

        epoch_outputs = []
        epoch_targets = []

    # Add progress bar for training
        train_loop = tqdm(train_dataloader, desc=f"🟦 Training Epoch {epoch+1}/{num_epochs}", leave=False)

        for waveforms, targets in train_loop:
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            # Normalize targets if specified
            if normalize_targets and target_stats is not None:
                targets = (targets - target_stats['mean']) / target_stats['std']

            input_values, attention_mask = preprocess_batch(waveforms, sampling_rate)
            input_values,  attention_mask = input_values.to(device), attention_mask.to(device)

            outputs = model(input_values, attention_mask=attention_mask)
            loss = criterion(outputs, targets)

            # Variance regularization
            pred_std = torch.std(outputs)
            target_std_batch = torch.std(targets)
            var_loss = torch.relu(0.3 * target_std_batch - pred_std) # Preds should cover a percentage of variance, atm = 30%

            # combine with main loss
            loss = loss + variance_reg_coeff * var_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # GRADIENT CLIPPING - helps with mode collapse
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_train_loss += loss.item()

            # Memory-efficient output diversity tracking
            batch_size = outputs.size(0)
            running_output_sum += outputs.sum().item()
            running_output_sq_sum += (outputs ** 2).sum().item()
            total_samples += batch_size

             # Update progress bar with current loss
            train_loop.set_postfix(loss=loss.item())

        # Calculate output statistics for mode collapse detection
        if total_samples > 0:
            output_mean = running_output_sum / total_samples
            output_var = (running_output_sq_sum / total_samples) - (output_mean ** 2)
            output_std = math.sqrt(max(output_var, 0))

          # Calculate target std for comparison
            all_targets = []
            for _, batch_targets in train_dataloader:
                if target_stats is not None:
                    batch_targets = (batch_targets - target_stats['mean']) / target_stats['std']
                all_targets.append(batch_targets)
            target_std = torch.cat(all_targets).std().item()

            if verbose and epoch % 2 == 0:
                print(f"Epoch {epoch+1}: Output std={output_std:.4f}, Target std={target_std:.4f}")

            # Improved mode collapse threshold - relative to target std
            mode_collapse_threshold = max(0.01, target_std * 0.05)  # 5% of target std
            if output_std < mode_collapse_threshold and epoch > 2:
                if verbose:
                    print(f"Mode collapse detected: (output_std={output_std:.4f}). Pruning trial.")
                if trial is not None:
                    trial.report(float("inf"), epoch)  # Report collapse to Optuna
                    raise optuna.exceptions.TrialPruned()  # Prune immediately
                else:
                  # Pad the remaining epochs
                  remaining_epochs = num_epochs - len(train_losses) - 1  # -1 because we haven't added current epoch yet
                  train_losses.append(float('inf'))
                  val_losses.append(float('inf'))
                  train_losses.extend([float('inf')] * remaining_epochs)
                  val_losses.extend([float('inf')] * remaining_epochs)
                  return train_losses, val_losses

        avg_train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        all_val_outputs = []
        all_val_targets = []

        with torch.no_grad():
            for waveforms, targets in val_dataloader:
                waveforms = waveforms.to(device)
                targets = targets.to(device)

                # Normalize targets if specified
                if normalize_targets and target_stats is not None:
                    targets = (targets - target_stats['mean']) / target_stats['std']

                input_values, attention_mask = preprocess_batch(waveforms, sampling_rate)
                input_values = input_values.to(device)
                attention_mask = attention_mask.to(device)

                outputs = model(input_values, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()

                # store for collapse monitoring
                all_val_outputs.append(outputs)
                all_val_targets.append(targets)

        avg_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Monitoring model collapse
        all_val_outputs = torch.cat(all_val_outputs)
        all_val_targets = torch.cat(all_val_targets)

        pred_std = torch.std(all_val_outputs).item()
        target_std = torch.std(all_val_targets).item()
        range_ratio = (all_val_outputs.max() - all_val_outputs.min()) / (all_val_targets.max() - all_val_targets.min() + 1e-8)

        # Compute R²
        val_preds_np = all_val_outputs.cpu().numpy()
        val_targets_np = all_val_targets.cpu().numpy()
        val_r2 = r2_score(val_targets_np, val_preds_np)

        # Logging learning rate and model collapse
        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, LR: {current_lr:.2e}")
            print(f"Validation Metrics - pred_std: {pred_std:.4f}, target_std: {target_std:.4f}, range_ratio: {range_ratio:.4f}")
            print(f"Validation R²: {val_r2:.4f}")

        # Optuna pruning
        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Early stopping measure based off validation loss
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        # Early stopping based off R2
        if val_r2 > best_val_r2 + min_delta:
            best_val_r2 = val_r2
            epochs_no_improve_r2 = 0
            # optionally save best model here
        else:
            epochs_no_improve_r2 += 1
            if epochs_no_improve_r2 >= patience:
                if verbose:
                    print(f"Early stopping on R² at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    return train_losses, val_losses

def test_model(
    model,
    test_dataloader,
    feature_extractor,
    criterion,
    device,
    sampling_rate,
    verbose=True):
    """
    Evaluate the model on a test set and return average test loss.
    """
    model.to(device)
    model.eval()

    running_test_loss = 0.0

    test_loop = tqdm(test_dataloader, desc="Testing", leave=False)

    with torch.no_grad():
        for waveforms, targets in test_loop:
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            input_values, attention_mask = preprocess_batch(waveforms, sampling_rate)
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_values, attention_mask=attention_mask)
            loss = criterion(outputs, targets)

            running_test_loss += loss.item()
            test_loop.set_postfix(loss=loss.item())

    avg_test_loss = running_test_loss / len(test_dataloader)

    if verbose:
        print(f"Test Loss: {avg_test_loss:.4f}")

    return avg_test_loss


def test_stats(
    model,
    test_dataloader,
    feature_extractor,
    criterion,
    device,
    sampling_rate,
    verbose=True):
    """
    Evaluates the model on test set, returns predictions and stats
    """
    model.to(device)
    model.eval()

    running_test_loss = 0.0

    all_preds = []
    all_targets = []

    test_loop = tqdm(test_dataloader, desc="Testing", leave=False)

    with torch.no_grad():
        for waveforms, targets in test_loop:
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            input_values, attention_mask = preprocess_batch(waveforms, sampling_rate)
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_values, attention_mask=attention_mask)

            # Squeezing for regression preds
            outsputs = outputs.squeeze()

            loss = criterion(outputs, targets)
            running_test_loss += loss.item()

            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

            test_loop.set_postfix(loss=loss.item())

    avg_test_loss = running_test_loss / len(test_dataloader)

 # Convert to arrays
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute regression metrics
    r2 = r2_score(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)

    if verbose:
        print(f"Test & Stats results for model {model}")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")

    return avg_test_loss, r2, mse, mae, all_preds, all_targets


def model_comparison(targets, preds, test_loss=None, r2=None, mse=None, mae=None, bins=30):
    """
    Shows model performance and
    Plots distribution comparison between targets and predictions
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Convert to numpy if tensors
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    if hasattr(preds, 'cpu'):
        preds = preds.cpu().numpy()

    # Flatten if needed
    targets = targets.flatten()
    preds = preds.flatten()

    # 1. Overlaid histograms
    axes[0, 0].hist(targets, bins=bins, alpha=0.7, label='Targets',
                    color='blue', density=True, edgecolor='black', linewidth=0.5)
    axes[0, 0].hist(preds, bins=bins, alpha=0.7, label='Predictions',
                    color='red', density=True, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Add statistics text
    target_stats = f'Targets: μ={np.mean(targets):.3f}, σ={np.std(targets):.3f}'
    pred_stats = f'Preds: μ={np.mean(preds):.3f}, σ={np.std(preds):.3f}'
    axes[0, 0].text(0.02, 0.98, f'{target_stats}\n{pred_stats}',
                    transform=axes[0, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 2. Box plots side by side
    box_data = [targets, preds]
    box_labels = ['Targets', 'Predictions']
    axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Box Plot Comparison')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Scatter plot (actual vs predicted)
    axes[1, 0].scatter(targets, preds, alpha=0.6, s=20)

    # Add perfect prediction line
    min_val = min(np.min(targets), np.min(preds))
    max_val = max(np.max(targets), np.max(preds))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--',
                    linewidth=2, label='Perfect prediction')

    axes[1, 0].set_xlabel('Target Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('Actual vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if r2 is not None:
        axes[1, 0].text(0.05, 0.95, f'R² = {r2:.4f}',
                        transform=axes[1, 0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=12, fontweight='bold')

    # 4. Q-Q plot
    stats.probplot(targets, plot=axes[1, 1], rvalue=True)
    axes[1, 1].get_lines()[0].set_markerfacecolor('blue')
    axes[1, 1].get_lines()[0].set_markeredgecolor('blue')
    axes[1, 1].get_lines()[0].set_label('Targets')

    # Add predictions to Q-Q plot
    stats.probplot(preds, plot=axes[1, 1], rvalue=True)
    axes[1, 1].get_lines()[2].set_markerfacecolor('red')
    axes[1, 1].get_lines()[2].set_markeredgecolor('red')
    axes[1, 1].get_lines()[2].set_label('Predictions')
    axes[1, 1].set_title('Q-Q Plot vs Normal Distribution')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Print test statistics if included
    if test_loss is not None or r2 is not None or mse is not None or mae is not None:
        print("\n" + "="*50)
        print("MODEL PERFORMANCE:")
        print("="*50)
        if test_loss is not None:
            print(f"Test Loss: {test_loss:.4f}")
        if r2 is not None:
            print(f"R² Score: {r2:.4f}")
        if mse is not None:
            print(f"MSE: {mse:.4f}")
        if mae is not None:
            print(f"MAE: {mae:.4f}")
        print("-" * 50)

    print("\n" + "="*50)
    print("DISTRIBUTION ANALYSIS")
    print("="*50)
    print(f"Targets    - Range: [{np.min(targets):.3f}, {np.max(targets):.3f}]")
    print(f"Predictions- Range: [{np.min(preds):.3f}, {np.max(preds):.3f}]")
    print(f"Target range span: {np.max(targets) - np.min(targets):.3f}")
    print(f"Prediction range span: {np.max(preds) - np.min(preds):.3f}")
    print(f"Range compression ratio: {(np.max(preds) - np.min(preds)) / (np.max(targets) - np.min(targets)):.3f}")

    print(f"\nTargets    - Mean: {np.mean(targets):.3f}, Std: {np.std(targets):.3f}")
    print(f"Predictions- Mean: {np.mean(preds):.3f}, Std: {np.std(preds):.3f}")

    # Percentile analysis
    target_percentiles = np.percentile(targets, [5, 25, 50, 75, 95])
    pred_percentiles = np.percentile(preds, [5, 25, 50, 75, 95])

    print(f"\nPercentiles (5th, 25th, 50th, 75th, 95th):")
    print(f"Targets:     {target_percentiles}")
    print(f"Predictions: {pred_percentiles}")