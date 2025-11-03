# Various dataset classes and functions needed to prep task and loaders

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class SingleLabelRegression(torch.utils.data.Dataset):
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


def collate_fn(batch):
    """
    Takes in waveform batch and adds padding
    Necessary because the waveforms have different dimensions
    """
    waveforms, labels = zip(*batch)  # unzip batch of tuples

    # Remove channel dim [1, T] → [T]
    waveforms = [w.squeeze(0) for w in waveforms]  # list of [T]

    # Pad sequence [B, T]
    waveforms_padded = pad_sequence(waveforms, batch_first=True)

    # Convert labels to tensor → [B]
    labels_tensor = torch.tensor(labels).float()  # [B]

    return waveforms_padded, labels_tensor


def split_data(
    waveforms,
    targets,
    target_name,
    batch_size=8,
    test_size=0.2,
    random_state=42,
    collate_fn=None,
    dataset_class=None
):
    """
    Split data into training, validation, and test sets for a single regression target
    (either valence or arousal).

    Args:
        waveforms (list of Tensors): list of waveform tensors
        targets (list of floats): target values (valence or arousal)
        target_name (str): name of the target ("valence" or "arousal"), used for print messages
        batch_size (int): batch size for DataLoader
        test_size (float): proportion of data for val+test combined (default = 0.3 → 70/15/15)
        random_state (int): random seed for reproducibility
        collate_fn (function): collate function for padding (required)
        dataset_class (torch Dataset class): dataset class to use (required)

    Returns:
        dict: containing DataLoaders for the selected target:
            {
                'train': DataLoader,
                'val': DataLoader,
                'test': DataLoader
            }
    """
    assert collate_fn is not None, "collate_fn must be provided."
    assert dataset_class is not None, "dataset_class must be provided."

    # === 1. Split indices ===
    train_indices, temp_indices = train_test_split(
        range(len(waveforms)),
        test_size=test_size,
        random_state=random_state
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=random_state
    )

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
    train_dataset = dataset_class(train_waveforms, train_targets)
    val_dataset = dataset_class(val_waveforms, val_targets)
    test_dataset = dataset_class(test_waveforms, test_targets)

    # === 5. Build dataloaders ===
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    }

    print(f"{target_name.capitalize()} dataloaders ready\n")
    return loaders