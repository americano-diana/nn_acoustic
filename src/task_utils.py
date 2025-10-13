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


def create_data_splits(
    waveforms,
    valences,
    arousals,
    batch_size=8,
    test_size=0.3,
    random_state=42,
    collate_fn=None,
    dataset_class=None
):
    """
    Split data into training, validation, and test sets for valence and arousal regression task
 
    Args:
        waveforms (list of Tensors): list of waveform tensors
        valences (list of floats): valence labels
        arousals (list of floats): arousal labels
        batch_size (int): batch size for DataLoader
        test_size (float): proportion of data for val+test combined (default = 0.3 → 70/15/15)
        random_state (int): random seed for reproducibility
        collate_fn (function): collate function for padding (required)
        dataset_class (torch Dataset class): dataset class to use (required)

    Returns:
        dict: containing DataLoaders for valence and arousal
            {
                'valence': {
                    'train': DataLoader,
                    'val': DataLoader,
                    'test': DataLoader
                },
                'arousal': {
                    'train': DataLoader,
                    'val': DataLoader,
                    'test': DataLoader
                }
            }
    """

    assert collate_fn is not None, "collate_fn must be provided."
    assert dataset_class is not None, "dataset_class must be provided."

    # === 1. Split indices ===
    train_indices, temp_indices = train_test_split(
        range(len(waveforms)),
        test_size=test_size,  # 30% for val + test
        random_state=random_state
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,  # split 30% evenly: 15% val / 15% test
        random_state=random_state
    )

    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")

    # === 2. Build subsets ===
    def subset_data(indices, data_list):
        return [data_list[i] for i in indices]

    # Training
    train_waveforms = subset_data(train_indices, waveforms)
    train_valences = subset_data(train_indices, valences)
    train_arousals = subset_data(train_indices, arousals)

    # Validation
    val_waveforms = subset_data(val_indices, waveforms)
    val_valences = subset_data(val_indices, valences)
    val_arousals = subset_data(val_indices, arousals)

    # Test
    test_waveforms = subset_data(test_indices, waveforms)
    test_valences = subset_data(test_indices, valences)
    test_arousals = subset_data(test_indices, arousals)

    # === 3. Build datasets ===
    valence_train_dataset = dataset_class(train_waveforms, train_valences)
    valence_val_dataset = dataset_class(val_waveforms, val_valences)
    valence_test_dataset = dataset_class(test_waveforms, test_valences)

    arousal_train_dataset = dataset_class(train_waveforms, train_arousals)
    arousal_val_dataset = dataset_class(val_waveforms, val_arousals)
    arousal_test_dataset = dataset_class(test_waveforms, test_arousals)

    # === 4. Build DataLoaders ===
    valence_loaders = {
        "train": DataLoader(valence_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        "val": DataLoader(valence_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(valence_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    }

    arousal_loaders = {
        "train": DataLoader(arousal_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        "val": DataLoader(arousal_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(arousal_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    }

    print("\n Data splitting complete")
    print("Loaders are:")
    print("  - valence_loaders['train'], valence_loaders['val'], valence_loaders['test']")
    print("  - arousal_loaders['train'], arousal_loaders['val'], arousal_loaders['test']")

    return {
        "valence": valence_loaders,
        "arousal": arousal_loaders
    }
