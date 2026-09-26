import os
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.utils import CFG


STATS_FILE = os.path.join(CFG['results_dir'], 'channel_stats.npy')


def compute_channel_stats(path, n=8000):
    """
    Compute per-channel mean and std from nonzero pixels only.
    Jet images are extremely sparse so using all pixels would
    pull the mean close to zero and make normalization useless.
    """
    with h5py.File(path, 'r') as f:
        imgs = f['jet'][:n].astype(np.float32)

    imgs = imgs.transpose(0, 3, 1, 2) / CFG['pixel_max']
    mean = np.zeros(8, dtype=np.float32)
    std  = np.zeros(8, dtype=np.float32)

    for c in range(8):
        ch = imgs[:, c].flatten()
        nz = ch[ch > 0]
        if len(nz) > 0:
            mean[c] = nz.mean()
            std[c]  = nz.std()
        else:
            mean[c], std[c] = 0.0, 1.0

    return mean, std


def load_channel_stats():
    if os.path.exists(STATS_FILE):
        data = np.load(STATS_FILE)
        return data[:8], data[8:]

    print("Channel stats not found, computing from training data...")
    mean, std = compute_channel_stats(CFG['train_path'])
    np.save(STATS_FILE, np.concatenate([mean, std]))
    print(f"Saved to {STATS_FILE}")
    return mean, std


chan_mean, chan_std = load_channel_stats()


class JetDataset(Dataset):
    def __init__(self, path, labelled=True, indices=None, verbose=True):
        self.labelled = labelled

        with h5py.File(path, 'r') as f:
            n   = f['jet'].shape[0]
            idx = np.sort(indices) if indices is not None else np.arange(n)
            self.imgs = f['jet'][idx]
            if labelled:
                self.labels = f['Y'][idx].flatten().astype(np.float32)

        if indices is not None:
            remap        = {old: new for new, old in enumerate(idx)}
            self.indices = np.array([remap[i] for i in indices])
        else:
            self.indices = np.arange(len(self.imgs))

        if verbose:
            print(f"  {path.split('/')[-1]} — {len(self.indices):,} samples  "
                  f"({self.imgs.nbytes / 1e6:.0f} MB)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ri  = int(self.indices[i])
        img = self.imgs[ri].astype(np.float32).transpose(2, 0, 1) / CFG['pixel_max']
        img = (img - chan_mean.reshape(8, 1, 1)) / (chan_std.reshape(8, 1, 1) + 1e-6)
        img = torch.tensor(img, dtype=torch.float32)
        img = F.pad(img, (0, 3, 0, 3), value=0.0)

        if not self.labelled:
            return img

        label = torch.tensor(self.labels[ri], dtype=torch.float32)
        mass  = img.relu().sum() / (128 * 128 * 8)
        return img, label, mass

def build_loaders(num_workers=4):
    """
    Returns train, val, test, pretrain loaders with a fixed 80/20 split.
    Val indices are held out before any training starts.
    """
    rng     = np.random.default_rng(CFG['seed'])
    all_idx = np.arange(8000)
    rng.shuffle(all_idx)

    val_idx   = all_idx[:CFG['val_size']]
    train_idx = all_idx[CFG['val_size']:]

    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0

    train_ds    = JetDataset(CFG['train_path'], labelled=True,  indices=train_idx)
    val_ds      = JetDataset(CFG['train_path'], labelled=True,  indices=val_idx)
    test_ds     = JetDataset(CFG['test_path'],  labelled=True)
    pretrain_ds = JetDataset(CFG['unlabelled_path'], labelled=False)

    train_loader = DataLoader(
        train_ds, batch_size=CFG['finetune_batch'],
        shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=512,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=512,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=CFG['pretrain_batch'],
        shuffle=True, num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, pretrain_loader