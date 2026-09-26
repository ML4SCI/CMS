"""
src/augmentations.py

Physics-aware augmentations for 8-channel CMS jet images.
Shared by SimCLR and DINO pretraining.

Included (detector symmetries):
    - horizontal / vertical flip  : phi and eta symmetry
    - 90-degree rotation          : jet rotational symmetry
    - channel dropout             : safe channels only, never Ch0/Ch3
                                    (Study 3 nb02: those collapse AUC to 0.47)
    - Gaussian noise σ ≤ 0.05    : within safe zone from Study 11a nb02

Excluded (physically meaningless):
    - random crop   : destroys jet substructure
    - color jitter  : not applicable to detector data
"""

import torch
import torch.nn.functional as F

# Ch0 and Ch3 are never dropped — zeroing either collapses AUC to chance
_SAFE_DROPOUT_CHANNELS = [1, 2, 4, 5, 6, 7]


class JetAugmentation:
    def __init__(self,
                 global_crop_size: int  = 128,
                 local_crop_size: int   = 96,
                 n_local_crops: int     = 4,
                 channel_drop_p: float  = 0.3,
                 noise_sigma_max: float = 0.05):
        self.global_size    = global_crop_size
        self.local_size     = local_crop_size
        self.n_local        = n_local_crops
        self.channel_drop_p = channel_drop_p
        self.noise_max      = noise_sigma_max

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > 0.5:
            img = img.flip(-1)

        if torch.rand(1).item() > 0.5:
            img = img.flip(-2)

        if torch.rand(1).item() > 0.5:
            k   = int(torch.randint(1, 4, (1,)).item())
            img = torch.rot90(img, k, dims=[-2, -1])

        if torch.rand(1).item() < self.channel_drop_p:
            ch      = _SAFE_DROPOUT_CHANNELS[
                int(torch.randint(0, len(_SAFE_DROPOUT_CHANNELS), (1,)).item())
            ]
            img     = img.clone()
            img[ch] = 0.0

        sigma = torch.rand(1).item() * self.noise_max
        if sigma > 0:
            img = img + torch.randn_like(img) * sigma

        return img

    def _global_crop(self, img: torch.Tensor) -> torch.Tensor:
        return self._augment(img)

    def _local_crop(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape
        size    = self.local_size
        top     = int(torch.randint(0, H - size + 1, (1,)).item())
        left    = int(torch.randint(0, W - size + 1, (1,)).item())
        crop    = img[:, top:top + size, left:left + size]
        # pad back to 128x128 so all views share the same spatial dim
        crop    = F.pad(crop, (0, W - size, 0, H - size), value=0.0)
        return self._augment(crop)

    def simclr_views(self, imgs: torch.Tensor):
        """Two independently augmented global views. Returns (view1, view2)."""
        imgs_cpu = imgs.cpu()
        B        = len(imgs_cpu)
        v1 = torch.stack([self._global_crop(imgs_cpu[i]) for i in range(B)])
        v2 = torch.stack([self._global_crop(imgs_cpu[i]) for i in range(B)])
        return v1.to(imgs.device), v2.to(imgs.device)

    def dino_views(self, imgs: torch.Tensor):
        """
        Multi-crop views for DINO.
        Returns global1, global2, [local_1, ..., local_n].
        Teacher sees only the two globals; student sees all.
        """
        imgs_cpu = imgs.cpu()
        B        = len(imgs_cpu)

        global1 = torch.stack([self._global_crop(imgs_cpu[i]) for i in range(B)])
        global2 = torch.stack([self._global_crop(imgs_cpu[i]) for i in range(B)])
        locals_ = [
            torch.stack([self._local_crop(imgs_cpu[i]) for i in range(B)])
            for _ in range(self.n_local)
        ]

        dev = imgs.device
        return global1.to(dev), global2.to(dev), [lc.to(dev) for lc in locals_]