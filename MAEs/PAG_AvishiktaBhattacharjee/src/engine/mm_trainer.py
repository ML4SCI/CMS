import os
from typing import List, Tuple, Dict, Callable, Optional, Union

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.distributed import all_gather, all_gather_object

from .trainer import Trainer
from ..utils import cleanup_ddp
from ..utils.data import JetClassDistributedSampler
from ..utils.viz import *


class MaskedModelTrainer(Trainer):
    """
    Trainer for model with one masked particle.

    Parameters
    ----------
    model: nn.Module
        The model to train.
    train_dataset: Dataset
        The dataset to use for training.
    val_dataset: Dataset
        The dataset to use for validation.
    test_dataset: Dataset, optional
        The dataset to use for testing.
    device: torch.device or int, optional
        Device to run the training on. Overrides config if provided.
    metric: Callable, optional
        A function to compute a metric for evaluation.
    config: TrainConfig, optional
        Configuration object containing training parameters.
    batch_size: int, optional
        Batch size for training. Overrides config if provided.
    criterion: Dict, optional
        Loss function configuration. Overrides config if provided.
    optimizer: Dict, optional
        Optimizer configuration. Overrides config if provided.
    optimizer_wrapper: Dict, optional
        Optimizer wrapper configuration. Overrides config if provided.
    scheduler: Dict, optional
        Learning rate scheduler configuration. Overrides config if provided.
    callbacks: List[Dict], optional
        A list of callbacks to execute during training. Overrides config if provided.
    num_epochs: int, optional
        Number of epochs to train for. Overrides config if provided.
    start_epoch: int, optional
        Epoch to start training from. Overrides config if provided.
    logging_dir: str, optional
        Directory to save logs. Overrides config if provided.
    logging_steps: int, optional
        Frequency of logging during training. Overrides config if provided.
    progress_bar: bool, optional
        Whether to display a tqdm progress bar. Useful to disable on HPC.
    save_best: bool, optional
        Whether to save the best model based on validation loss. Overrides config if provided.
    save_ckpt: bool, optional
        Whether to save checkpoints during training. Overrides config if provided.
    save_fig: bool, optional
        Whether to save evaluation figures. Overrides config if provided.
    num_workers: int, optional
        Number of workers for data loading. Overrides config if provided.
    pin_memory: bool, optional
        Whether to use pinned memory for data loading. Overrides config if provided.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {
            'epoch': [], 
            'pT_loss': [],
            'eta_loss': [],
            'phi_loss': [],
            'energy_loss': [],
            'val_loss': []
        }

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        try:
            # Callback before training
            for cb in self.callbacks:
                cb.on_train_begin(trainer=self)

            total_steps = self.num_epochs * len(self.train_loader)
            start_step = self.start_epoch * len(self.train_loader)
            if self.progress_bar and self.rank == 0:
                global_bar = tqdm(
                    total=total_steps,
                    initial=start_step,
                    desc="Training",
                    dynamic_ncols=True
                )
            else:
                class _NoOpBar:
                    def set_postfix(self, *args, **kwargs): pass
                    def update(self, *args, **kwargs): pass
                global_bar = _NoOpBar()

            for epoch in range(self.start_epoch, self.num_epochs):
                # Make DistributedSampler shuffle with a different seed each epoch
                if self._is_distributed and isinstance(self.train_loader.batch_sampler, JetClassDistributedSampler):
                    self.train_loader.batch_sampler.set_epoch(epoch)

                # Callback at the beginning of each epoch
                for cb in self.callbacks:
                    cb.on_epoch_begin(epoch, trainer=self)

                # Training phase
                self.model.train()
                running_loss_sum = 0.0
                loss_components_sum = None
                running_count = 0

                for batch_idx, (X, y, mask_idx) in enumerate(self.train_loader):
                    step = epoch * len(self.train_loader) + batch_idx + 1

                    X = X.to(self.device, non_blocking=self.pin_memory)
                    y = y.to(self.device, non_blocking=self.pin_memory)
                    mask_idx = mask_idx.to(self.device).long()

                    self.optimizer.zero_grad()
                    outputs = self.model(X, mask_idx)
                    loss, components = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()
                    bsz = y.size(0)
                    running_loss_sum += float(loss.item()) * bsz
                    running_count += bsz

                    if loss_components_sum is None:
                        loss_components_sum = [0.0 for _ in components]
                    for i, comp in enumerate(components):
                        loss_components_sum[i] += float(comp.item()) * bsz

                    avg_loss = running_loss_sum / running_count
                    avg_components = [s / running_count for s in loss_components_sum]

                    # Short summary
                    if self.rank == 0:
                        if step % self.logging_steps == 0 or step == total_steps:
                            tqdm.write(
                                f"step: {step}/{total_steps} | "
                                f"pT_loss: {avg_components[0]:.4f} | "
                                f"eta_loss: {avg_components[1]:.4f} | "
                                f"phi_loss: {avg_components[2]:.4f} | "
                                f"energy_loss: {avg_components[3]:.4f} | "
                                f"total_train_loss: {avg_loss:.4f}"
                            )

                        global_bar.set_postfix({
                            "epoch": f"{epoch + 1}/{self.num_epochs}",
                            "avg_loss": f"{avg_loss:.4f}"
                        })

                    global_bar.update(1)

                # Validation phase
                self.model.eval()
                val_loss_sum = 0.0
                val_comp_sum = None
                val_count = 0

                with torch.no_grad():
                    for X_val, y_val, mask_idx in self.val_loader:
                        X_val = X_val.to(self.device, non_blocking=self.pin_memory)
                        y_val = y_val.to(self.device, non_blocking=self.pin_memory)
                        mask_idx = mask_idx.to(self.device).long()

                        outputs_val = self.model(X_val, mask_idx)
                        loss_val, v_components = self.criterion(outputs_val, y_val)
                        bsz = y_val.size(0)
                        val_loss_sum += float(loss_val.item()) * bsz

                        if val_comp_sum is None:
                            val_comp_sum = torch.zeros(
                                len(v_components),
                                dtype=torch.float64,
                                device=self.device
                            )

                        val_comp_sum += torch.as_tensor(
                            v_components,
                            dtype=torch.float64,
                            device=self.device
                        ) * bsz
                        val_count += bsz

                # Gather validation results from all processes
                if self._is_distributed:
                    pack = torch.tensor([val_loss_sum, float(val_count)], dtype=torch.float64, device=self.device)
                    packs = [torch.zeros_like(pack) for _ in range(self.world_size)]
                    comps = [torch.zeros_like(val_comp_sum) for _ in range(self.world_size)]
                    all_gather(packs, pack)
                    all_gather(comps, val_comp_sum)

                    total_loss_sum = sum(p[0].item() for p in packs)
                    total_count = int(sum(p[1].item() for p in packs))
                    total_comp_sum = torch.stack(comps, dim=0).sum(dim=0)
                else:
                    total_loss_sum = val_loss_sum
                    total_count = val_count
                    total_comp_sum = val_comp_sum

                # Global averages for validation
                val_loss = total_loss_sum / max(total_count, 1)
                avg_val_components = (total_comp_sum / max(total_count, 1)).tolist()

                # Short summary for validation
                if self.rank == 0:
                    tqdm.write(
                        f"epoch: {epoch + 1}/{self.num_epochs} | "
                        f"pT_loss: {avg_val_components[0]:.4f} | "
                        f"eta_loss: {avg_val_components[1]:.4f} | "
                        f"phi_loss: {avg_val_components[2]:.4f} | "
                        f"energy_loss: {avg_val_components[3]:.4f} | "
                        f"total_val_loss: {val_loss:.4f}"
                    )

                if self.scheduler:
                    self.scheduler.step()

                # Get learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Save best model
                if self.best_model_path and self.rank == 0 and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    to_save = self.model.module if self._is_distributed else self.model
                    torch.save(to_save.state_dict(), self.best_model_path)

                # Update history
                self.history['epoch'].append(epoch + 1)
                self.history['pT_loss'].append(avg_components[0])
                self.history['eta_loss'].append(avg_components[1])
                self.history['phi_loss'].append(avg_components[2])
                self.history['energy_loss'].append(avg_components[3])
                self.history['val_loss'].append(val_loss)

                # Save a checkpoint every epoch
                self.save_checkpoint(epoch)

                # Log results
                logs = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr,
                }
                self.log_csv(logs)

                # Callback after each epoch
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch, trainer=self, logs=logs)

                # Break if any callback says to stop
                if any(getattr(cb, 'should_stop', False) for cb in self.callbacks):
                    break

            # Callback after training
            for cb in self.callbacks:
                cb.on_train_end(trainer=self)
        except KeyboardInterrupt:
            if self.rank == 0:
                print(f"\nTraining interrupted at epoch {epoch + 1}.")

            cleanup_ddp()

        return self.history, self.model

    @torch.no_grad()
    def evaluate(
        self,
        plot: Optional[Union[Callable, List[Callable]]] = None
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        if self.test_loader is None:
            raise ValueError("Test dataset is not provided.")
        
        self.model.eval()
        loss_sum = 0.0
        metric_sum = 0.0
        count = 0
        comp_sum = None
        local_true = []
        local_pred = []

        for X_test, y_test, mask_idx in self.test_loader:
            X_test = X_test.to(self.device, non_blocking=self.pin_memory)
            y_test = y_test.to(self.device, non_blocking=self.pin_memory)
            mask_idx = mask_idx.to(self.device).long()

            outputs_test = self.model(X_test, mask_idx)

            # Guarantee phi to be in [-pi, pi]
            outputs_test[..., 2] = torch.remainder(outputs_test[..., 2] + torch.pi, 2 * torch.pi) - torch.pi
            
            tloss, t_components = self.criterion(outputs_test, y_test)
            bsz = y_test.size(0)
            loss_sum += float(tloss.item()) * bsz

            if self.metric:
                metric_sum += float(self.metric(outputs_test, y_test)) * bsz

            if comp_sum is None:
                comp_sum = torch.zeros(
                    len(t_components),
                    dtype=torch.float64,
                    device=self.device
                )

            comp_sum += torch.as_tensor(
                t_components,
                dtype=torch.float64,
                device=self.device
            ) * bsz
            count += bsz

            local_true.append(y_test.cpu().numpy())
            local_pred.append(outputs_test.cpu().numpy())

        # Stack per-rank arrays
        y_true_local = np.concatenate(local_true, axis=0) if local_true else np.empty((0,))
        y_pred_local = np.concatenate(local_pred, axis=0) if local_pred else np.empty((0,))

        # Gather scalars from all processes
        if self._is_distributed:
            pack = torch.tensor([loss_sum, metric_sum, float(count)], dtype=torch.float64, device=self.device)
            packs = [torch.zeros_like(pack) for _ in range(self.world_size)]
            comps = [torch.zeros_like(comp_sum) for _ in range(self.world_size)]
            all_gather(packs, pack)
            all_gather(comps, comp_sum)

            total_loss_sum = sum(p[0].item() for p in packs)
            total_metric_sum = sum(p[1].item() for p in packs)
            total_count = int(sum(p[2].item() for p in packs))
            total_comp_sum = torch.stack(comps, dim=0).sum(dim=0)
        else:
            total_loss_sum = loss_sum
            total_metric_sum = metric_sum
            total_count = count
            total_comp_sum = comp_sum

        # Global averages
        test_loss = total_loss_sum / max(total_count, 1)
        test_metric = (total_metric_sum / max(total_count, 1)) if self.metric else 0.0
        avg_test_components = (total_comp_sum / max(total_count, 1)).tolist()

        # Gather variable-length arrays from all processes
        if self._is_distributed:
            gathered_true = [None] * self.world_size
            gathered_pred = [None] * self.world_size
            all_gather_object(gathered_true, y_true_local)
            all_gather_object(gathered_pred, y_pred_local)

            if self.rank == 0:
                y_true = np.concatenate(gathered_true, axis=0) if gathered_true else np.empty((0,))
                y_pred = np.concatenate(gathered_pred, axis=0) if gathered_pred else np.empty((0,))
            else:
                y_true, y_pred = y_true_local, y_pred_local
        else:
            y_true, y_pred = y_true_local, y_pred_local

        if self.rank == 0:
            print(
                f"test_loss: {test_loss:.4f} | "
                f"pT_loss: {avg_test_components[0]:.4f} | "
                f"eta_loss: {avg_test_components[1]:.4f} | "
                f"phi_loss: {avg_test_components[2]:.4f} | "
                f"energy_loss: {avg_test_components[3]:.4f}"
            )

            # Visualization
            if plot is not None:
                if isinstance(plot, list):
                    for i, viz in enumerate(plot):
                        output_path = os.path.join(self.outputs_dir, f"{self.run_name}_viz_{i + 1}.png")
                        output_path = output_path if self.save_fig else None
                        viz(y_true, y_pred, save_fig=output_path)
                else:
                    output_path = os.path.join(self.outputs_dir, f"{self.run_name}_particle_reconstruction.png")
                    output_path = output_path if self.save_fig else None
                    plot(y_true, y_pred, save_fig=output_path)

        return test_loss, test_metric, y_true, y_pred