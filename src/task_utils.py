import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
   
class SingleRegression(torch.utils.data.Dataset):
    """
    Creates a dataset class called "SingleLabelRegression" that takes in the waveforms and one chosen target (either valence or arousal)
    """
    def __init__(self, waveforms, targets):
        self.waveforms = waveforms
        self.targets = targets  # Single label (valence or arousal)

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        return self.waveforms[idx], torch.tensor(self.targets[idx], dtype=torch.float32)


class FlexibleRegression(torch.utils.data.Dataset):
    """
    A toggleable dataset for both single-label and double-label regression.

    If targets are 1D -> Single label (e.g., valence)
    If targets are 2D -> Double label (e.g., [valence, arousal])
    Or you can explicitly set num_outputs for clarity.
    """
    def __init__(self, waveforms, targets, num_outputs=None):
        """
        Args:
            waveforms: list/tensor of shape (N, ...)
            targets: list/array/tensor of shape (N,) or (N, 2)
            num_outputs: optional int (1 or 2). If None, it will auto-detect.
        """
        self.waveforms = waveforms
        self.targets = targets

        # Auto-detect output dimensionality if not provided
        if num_outputs is None:
            # handle both list-of-lists and tensors
            first_target = targets[0]
            self.num_outputs = len(first_target) if hasattr(first_target, "__len__") and not isinstance(first_target, (str, bytes)) else 1
        else:
            self.num_outputs = num_outputs

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        # Ensure correct shape: scalar for single output, vector for double
        if self.num_outputs == 1:
            target = target.squeeze()  # shape ()
        return waveform, target
    
def collate_fn(batch):
    """
    Collate function for regression (single or double output).

    Pads variable-length waveforms and stacks labels automatically.
    Supports:
      - Single-label regression: labels → shape [B]
      - Double-label regression: labels → shape [B, 2]
    """
    waveforms, labels = zip(*batch)  # unzip list of (waveform, label)

    # Remove channel dim [1, T] → [T]
    waveforms = [w.squeeze(0) for w in waveforms]

    # Pad sequences to the same length
    waveforms_padded = pad_sequence(waveforms, batch_first=True)  # shape [B, T]

    # Handle label shapes
    # If labels are scalars → [B]
    # If labels are vectors (e.g., [valence, arousal]) → [B, 2]
    if isinstance(labels[0], (list, tuple, torch.Tensor)) and len(torch.tensor(labels[0]).shape) > 0:
        labels_tensor = torch.stack([torch.tensor(l, dtype=torch.float32) for l in labels])  # [B, 2]
    else:
        labels_tensor = torch.tensor(labels, dtype=torch.float32)  # [B]

    return waveforms_padded, labels_tensor

def split_data(
    waveforms,
    targets,
    target_name,
    batch_size=8,
    test_size=0.2,
    random_state=42,
    collate_fn=None,
    dataset_class=None,
    num_outputs=1
):
    """
    Split data into training, validation, and test sets for regression tasks.

    Supports:
      - Single-label regression (valence OR arousal)
      - Double-label regression (valence + arousal)

    Args:
        waveforms (list of Tensors): waveform tensors
        targets (list or array): target values, shape (N,) or (N, 2)
        target_name (str): descriptive name ("valence", "arousal", or "both")
        batch_size (int): batch size for DataLoader
        test_size (float): fraction of data for val+test
        random_state (int): random seed
        collate_fn (function): required for Wav2Vec2 padding
        dataset_class (torch Dataset class): e.g., SingleRegression or FlexibleRegression
        num_outputs (int): 1 for single regression, 2 for double regression

    Returns:
        dict of DataLoaders for train, val, and test splits.
    """
    assert collate_fn is not None, "collate_fn must be provided."
    assert dataset_class is not None, "dataset_class must be provided."

    # === 1. Split indices ===
    train_indices, temp_indices = train_test_split(
        range(len(waveforms)), test_size=test_size, random_state=random_state
    )
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=random_state)

    print(f"🔹 {target_name.upper()} Data Split:")
    print(f"  Training samples:   {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    print(f"  Test samples:       {len(test_indices)}")

    # === 2. Helper for subset selection ===
    def subset(indices, data):
        return [data[i] for i in indices]

    # === 3. Create subsets ===
    train_waveforms = subset(train_indices, waveforms)
    val_waveforms = subset(val_indices, waveforms)
    test_waveforms = subset(test_indices, waveforms)

    train_targets = subset(train_indices, targets)
    val_targets = subset(val_indices, targets)
    test_targets = subset(test_indices, targets)

    # === 4. Build datasets ===
    if num_outputs == 2:
        # For valence + arousal (double regression)
        train_dataset = dataset_class(train_waveforms, train_targets, num_outputs=2)
        val_dataset = dataset_class(val_waveforms, val_targets, num_outputs=2)
        test_dataset = dataset_class(test_waveforms, test_targets, num_outputs=2)
    else:
        # For single regression
        train_dataset = dataset_class(train_waveforms, train_targets)
        val_dataset = dataset_class(val_waveforms, val_targets)
        test_dataset = dataset_class(test_waveforms, test_targets)

    # === 5. Build DataLoaders ===
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    }

    print(f"{target_name.capitalize()} dataloaders ready (num_outputs={num_outputs})\n")
    return loaders