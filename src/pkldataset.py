import os
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.ndimage

# -------------------------------------------------------------------------
# Label Mappings
# -------------------------------------------------------------------------
class_to_onehot = {
    'R05': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'R10': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'R15': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'R20': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'R25': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'R30': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'R35': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'R40': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'R45': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'R50': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

thickness_to_id = {
    'T185': 0,
    'T188': 1,
    'T191': 2,
    'T194': 3,
    'T197': 4
}

thickness_to_val = {
    'T185': 1.85,
    'T188': 1.88,
    'T191': 1.91,
    'T194': 1.94,
    'T197': 1.97
}

condition_to_id = {
    'HC': 0,
    'DC': 1
}


class PKLDataset(Dataset):
    def __init__(self, folder_path, modality="fusion", task="wear", transform_type=None):
        """
        Args:
            folder_path (str, Path or list): Path or list of paths with .pkl files.
            modality (str): 'single' (1 channel) or 'fusion' (RP + AZ, 2 channels). Default: 'fusion'.
            task (str):
                - 'wear': Returns one-hot tensor (10,) for R05..R50. (Default, backwards-compatible).
                - 'thickness': Returns integer class tensor 0..4 for T185..T197.
                - 'condition': Returns binary class tensor 0 (HC) or 1 (DC).
                - 'multitask': Returns (data_tensor, {'wear': ..., 'thickness': ..., 'condition': ...}).
            transform_type (str, optional): Optional signal transformation.
        """
        self.modality = modality.lower()
        self.task = task.lower()
        self.transform_type = transform_type

        if isinstance(folder_path, (str, Path)):
            self.folder_paths = [Path(folder_path)]
        else:
            self.folder_paths = [Path(p) for p in folder_path]

        # Flat sample index: stores (base_folder, filename, condition_id, thickness_id)
        self.samples = []
        for f_path in self.folder_paths:
            parts = f_path.resolve().parts

            # Detect Condition (HC vs DC)
            if "HC" in parts:
                cond_id = condition_to_id['HC']
            elif "DC" in parts:
                cond_id = condition_to_id['DC']
            else:
                cond_id = -1

            # Detect Thickness (T185..T197)
            t_id = -1
            for t_key in thickness_to_id:
                if t_key in parts:
                    t_id = thickness_to_id[t_key]
                    break

            pkl_files = [f for f in os.listdir(f_path) if f.endswith('.pkl')]
            pkl_files.sort()
            for fname in pkl_files:
                self.samples.append((f_path, fname, cond_id, t_id))

        # Backward compatibility with scripts inspecting dataset.files
        self.files = [s[1] for s in self.samples]
        self.folder_path = str(self.folder_paths[0]) if len(self.folder_paths) == 1 else ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_folder, filename, cond_id, t_id = self.samples[idx]

        # -------------------------------------------------------------
        # 1. Load sensor data
        # -------------------------------------------------------------
        main_path = base_folder / filename
        with open(main_path, 'rb') as f:
            main_data = pickle.load(f)

        if self.modality == 'single':
            data_np = np.expand_dims(main_data, axis=0)

        elif self.modality == 'fusion':
            sensor_name = base_folder.name.upper()

            if sensor_name == "RP":
                paired_folder = base_folder.parent / "AZ"
                paired_path = paired_folder / filename
                with open(paired_path, 'rb') as f:
                    paired_data = pickle.load(f)
                data_np = np.stack([main_data, paired_data], axis=0)

            elif sensor_name == "AZ":
                paired_folder = base_folder.parent / "RP"
                paired_path = paired_folder / filename
                with open(paired_path, 'rb') as f:
                    paired_data = pickle.load(f)
                data_np = np.stack([paired_data, main_data], axis=0)
            else:
                raise ValueError(f"For 'fusion', the leaf directory must be 'RP' or 'AZ'. Got: {base_folder}")
        else:
            raise ValueError(f"Unknown modality: '{self.modality}'. Expected 'single' or 'fusion'.")

        data_tensor = torch.tensor(data_np, dtype=torch.float32)

        # -------------------------------------------------------------
        # 2. Extract targets
        # -------------------------------------------------------------
        class_label = filename.split('_')[0]
        wear_vector = class_to_onehot[class_label]
        wear_tensor = torch.tensor(wear_vector, dtype=torch.float32)

        if self.task == 'wear':
            return data_tensor, wear_tensor

        elif self.task == 'thickness':
            if t_id == -1:
                raise ValueError(f"Could not infer sheet thickness from path: {base_folder}")
            thickness_tensor = torch.tensor(t_id, dtype=torch.long)
            return data_tensor, thickness_tensor

        elif self.task == 'condition':
            if cond_id == -1:
                raise ValueError(f"Could not infer condition (HC/DC) from path: {base_folder}")
            condition_tensor = torch.tensor(cond_id, dtype=torch.long)
            return data_tensor, condition_tensor

        elif self.task in ('multitask', 'all'):
            targets = {
                "wear": wear_tensor,
                "thickness": torch.tensor(t_id, dtype=torch.long),
                "condition": torch.tensor(cond_id, dtype=torch.long)
            }
            return data_tensor, targets

        else:
            raise ValueError(f"Unknown task: '{self.task}'. Choose from 'wear', 'thickness', 'condition', 'multitask'.")

    @staticmethod
    def split_dataset(folder_path, modality="fusion", task="wear", transform_type=None, train_ratio=0.8, random_seed=42):
        full_dataset = PKLDataset(folder_path=folder_path, modality=modality, task=task, transform_type=transform_type)
        train_size = int(len(full_dataset) * train_ratio)
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        return train_dataset, val_dataset