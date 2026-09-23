from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def load_npz_splits(npz_path: str, split_seed: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a pre-serialised JetClass archive and split it 80/10/10 (stratified), as in the PAG notebook.

    Parameters
    ----------
    npz_path: str
        Archive with 'X_particles' of shape (num_jets, 4, max_num_particles) and one-hot 'Y' of shape (num_jets, 10).
    split_seed: int, optional
        Random state of both splits. Keep it fixed so training and evaluation see the same test set.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        {'train': (X, y), 'val': (X, y), 'test': (X, y)}
    """
    archive = np.load(npz_path, mmap_mode='r')
    X_particles = archive['X_particles']
    y = archive['Y']

    X_train, X_val, y_train, y_val = train_test_split(
        X_particles, y, test_size=0.2, random_state=split_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=split_seed, stratify=y_val
    )

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
