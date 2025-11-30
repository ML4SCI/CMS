import os
import csv
from typing import List, Tuple, Dict, Callable, Optional, Union
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import (
    is_available, is_initialized,
    get_rank, get_world_size,
    all_gather, all_gather_object
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from ..configs import TrainConfig
from ..loss import LOSS_REGISTRY
from ..optim import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from ..utils import (
    CALLBACK_REGISTRY,
    get_loss_from_config,
    get_optim_from_config,
    get_optim_wrapper_from_config,
    get_scheduler_from_config,
    get_callbacks_from_config,
    cleanup_ddp
)
from ..utils.data import JetClassDistributedSampler
from ..utils.viz import *


class Trainer:
    """
    Base class for training models.
    
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
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        device: Optional[Union[torch.device, int]] = None,
        metric: Optional[Callable] = None,
        config: Optional[TrainConfig] = None,
        # Parameters below can override config if supplied explicitly
        batch_size: Optional[int] = None,
        criterion: Optional[Dict] = None,
        optimizer: Optional[Dict] = None,
        optimizer_wrapper: Optional[Dict] = None,
        scheduler: Optional[Dict] = None,
        callbacks: Optional[List[Dict]] = None,
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = None,
        logging_dir: Optional[str] = None,
        logging_steps: Optional[int] = None,
        progress_bar: Optional[bool] = None,
        save_best: Optional[bool] = None,
        save_ckpt: Optional[bool] = None,
        save_fig: Optional[bool] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None
    ):
        self.rank = 0
        self.world_size = 1

        # Prepare the device for multi-GPU training if available
        if is_available() and is_initialized():
            self.rank = get_rank()
            self.world_size = get_world_size()
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f'cuda:{self.rank}')
            else:
                self.device = torch.device('cpu')

        # Prepare the model for multi-GPU training if available
        self.model = model.to(self.device)
        self._is_distributed = self.world_size > 1
        if self._is_distributed:
            self.model = DDP(
                module=self.model,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=True,
                gradient_as_bucket_view=True
            )

        # Use config if provided, otherwise use defaults
        if config is not None:
            self.batch_size = batch_size if batch_size is not None else config.batch_size
            self.criterion = get_loss_from_config(criterion if criterion is not None else config.criterion, LOSS_REGISTRY)
            self.optimizer = get_optim_from_config(optimizer if optimizer is not None else config.optimizer, OPTIM_REGISTRY, self.model)
            if optimizer_wrapper is not None or config.optimizer_wrapper is not None:
                self.optimizer = get_optim_wrapper_from_config(optimizer_wrapper if optimizer_wrapper is not None else config.optimizer_wrapper, OPTIM_REGISTRY, self.optimizer)
            self.scheduler = get_scheduler_from_config(scheduler if scheduler is not None else config.scheduler, SCHEDULER_REGISTRY, self.optimizer)
            self.callbacks = get_callbacks_from_config(callbacks if callbacks is not None else config.callbacks, CALLBACK_REGISTRY)
            self.num_epochs = num_epochs if num_epochs is not None else config.num_epochs
            self.start_epoch = start_epoch if start_epoch is not None else config.start_epoch
            self.logging_dir = logging_dir if logging_dir is not None else config.logging_dir
            self.logging_steps = logging_steps if logging_steps is not None else config.logging_steps
            self.progress_bar = progress_bar if progress_bar is not None else config.progress_bar
            self.save_best = save_best if save_best is not None else config.save_best
            self.save_ckpt = save_ckpt if save_ckpt is not None else config.save_ckpt
            self.save_fig = save_fig if save_fig is not None else config.save_fig
            self.num_workers = num_workers if num_workers is not None else config.num_workers
            self.pin_memory = pin_memory if pin_memory is not None else config.pin_memory
        else:
            self.batch_size = batch_size if batch_size is not None else 64
            if criterion is None:
                raise ValueError("Criterion must be provided if config is not supplied.")
            self.criterion = get_loss_from_config(criterion, LOSS_REGISTRY)
            if optimizer is None:
                raise ValueError("Optimizer must be provided if config is not supplied.")
            self.optimizer = get_optim_from_config(optimizer, OPTIM_REGISTRY, self.model)
            if optimizer_wrapper is not None:
                self.optimizer = get_optim_wrapper_from_config(optimizer_wrapper, OPTIM_REGISTRY, self.optimizer)
            self.scheduler = get_scheduler_from_config(scheduler, SCHEDULER_REGISTRY, self.optimizer) if scheduler else None
            self.callbacks = get_callbacks_from_config(callbacks, CALLBACK_REGISTRY) if callbacks is not None else None
            self.num_epochs = num_epochs if num_epochs is not None else 20
            self.start_epoch = start_epoch if start_epoch is not None else 0
            self.logging_dir = logging_dir if logging_dir is not None else 'logs'
            self.logging_steps = logging_steps if logging_steps is not None else 25
            self.progress_bar = progress_bar if progress_bar is not None else True
            self.save_best = save_best if save_best is not None else True
            self.save_ckpt = save_ckpt if save_ckpt is not None else True
            self.save_fig = save_fig if save_fig is not None else False
            self.num_workers = num_workers if num_workers is not None else 0
            self.pin_memory = pin_memory if pin_memory is not None else False
        
        # Initialize data samplers for distributed training
        train_sampler = JetClassDistributedSampler(
            files_by_class=train_dataset.files_by_class,
            events_per_file=train_dataset.events_per_file,
            batch_size=self.batch_size,
            rank=self.rank,
            world_size=self.world_size,
            shuffle_files=True
        ) if self._is_distributed else None
        val_sampler = JetClassDistributedSampler(
            files_by_class=val_dataset.files_by_class,
            events_per_file=val_dataset.events_per_file,
            batch_size=self.batch_size,
            rank=self.rank,
            world_size=self.world_size,
            shuffle_files=False
        ) if self._is_distributed else None
        test_sampler = JetClassDistributedSampler(
            files_by_class=test_dataset.files_by_class,
            events_per_file=test_dataset.events_per_file,
            batch_size=self.batch_size,
            rank=self.rank,
            world_size=self.world_size,
            shuffle_files=False
        ) if (test_dataset is not None and self._is_distributed) else None

        # Initialize data loaders
        if self._is_distributed:
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_sampler=train_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_sampler=test_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            ) if test_dataset is not None else None
        else:
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            ) if test_dataset is not None else None

        # Initialize metrics and history
        self.metric = metric
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_metric': [],
            'val_loss': [],
            'val_metric': []
        }
        self.best_val_loss = min(self.history['val_loss']) if self.history['val_loss'] else float('inf')

        # Initialize the logging directory
        if isinstance(self.model, DDP):
            self.model_name = self.model.module.__class__.__name__
        else:
            self.model_name = self.model.__class__.__name__
            
        os.makedirs(self.logging_dir, exist_ok=True)
        self.log_dir = os.path.join(self.logging_dir, self.model_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Subfolders
        self.best_models_dir = os.path.join(self.log_dir, 'best')
        self.checkpoints_dir = os.path.join(self.log_dir, 'checkpoints')
        self.loggings_dir = os.path.join(self.log_dir, 'logging')
        self.outputs_dir = os.path.join(self.log_dir, 'output')
        os.makedirs(self.best_models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.loggings_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Determine run index
        self.run_name = self._get_next_run_index()

        # Logging and best model paths
        self._log_header_written = False
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.run_name}.pt") if self.save_ckpt else None
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")

    def _get_next_run_index(self) -> str:
        return f"pid{os.getpid()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}"

    def _set_logging_paths(self, run_name: str):
        self.run_name = run_name
        self._log_header_written = True
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.run_name}.pt") if self.save_ckpt else None
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")

    def save_checkpoint(self, epoch: int):
        model_state = self.model.module.state_dict() if self._is_distributed else self.model.state_dict()
        if self.checkpoint_path and self.rank == 0:
            checkpoint = {
                'run_name': self.run_name,
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'history': self.history
            }
            torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._set_logging_paths(checkpoint['run_name'])
        target = self.model.module if self._is_distributed else self.model
        target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.history = checkpoint['history']
    
    def load_best_model(self, best_model_path: str):
        run_name = os.path.splitext(os.path.basename(best_model_path))[0]
        self._set_logging_paths(run_name)
        target = self.model.module if self._is_distributed else self.model
        target.load_state_dict(torch.load(best_model_path, map_location=self.device))

    def log_csv(self, log_dict: Dict[str, float]):
        if self.rank != 0:
            return
        
        write_header = not self._log_header_written
        with open(self.logging_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
            if write_header:
                writer.writeheader()
                self._log_header_written = True

            writer.writerow(log_dict)

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
                running_metric_sum = 0.0
                running_count = 0

                for batch_idx, (X, y) in enumerate(self.train_loader):
                    step = epoch * len(self.train_loader) + batch_idx + 1

                    X = X.to(self.device, non_blocking=self.pin_memory)
                    y = y.to(self.device, non_blocking=self.pin_memory)

                    self.optimizer.zero_grad()
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()
                    bsz = y.size(0)
                    running_loss_sum += float(loss.item()) * bsz
                    
                    if self.metric:
                        running_metric_sum += float(self.metric(outputs, y)) * bsz

                    running_count += bsz

                    avg_loss = running_loss_sum / running_count
                    avg_metric = (running_metric_sum / running_count) if self.metric else 0.0

                    # Short summary
                    if self.rank == 0:
                        if step % self.logging_steps == 0 or step == total_steps:
                            tqdm.write(
                                f"step: {step}/{total_steps} | "
                                f"train_loss: {avg_loss:.4f} | "
                                f"train_metric: {avg_metric:.4f}"
                            )

                        global_bar.set_postfix({
                            "epoch": f"{epoch + 1}/{self.num_epochs}",
                            "avg_loss": f"{avg_loss:.4f}",
                            "avg_metric": f"{avg_metric:.4f}"
                        })

                    global_bar.update(1)

                # Validation phase
                self.model.eval()
                val_loss_sum = 0.0
                val_metric_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    for X_val, y_val in self.val_loader:
                        X_val = X_val.to(self.device, non_blocking=self.pin_memory)
                        y_val = y_val.to(self.device, non_blocking=self.pin_memory)

                        outputs_val = self.model(X_val)
                        loss_val = self.criterion(outputs_val, y_val)
                        bsz = y_val.size(0)
                        val_loss_sum += float(loss_val.item()) * bsz

                        if self.metric:
                            val_metric_sum += float(self.metric(outputs_val, y_val)) * bsz

                        val_count += bsz

                # Gather validation results from all processes
                if self._is_distributed:
                    packed = torch.tensor(
                        data=[val_loss_sum, val_metric_sum, float(val_count)],
                        dtype=torch.float64,
                        device=self.device,
                    )
                    gathered = [torch.zeros_like(packed) for _ in range(self.world_size)]
                    all_gather(gathered, packed)

                    total_loss_sum = sum(g[0].item() for g in gathered)
                    total_metric_sum = sum(g[1].item() for g in gathered)
                    total_count = int(sum(g[2].item() for g in gathered))
                else:
                    total_loss_sum = val_loss_sum
                    total_metric_sum = val_metric_sum
                    total_count = val_count

                # Global averages
                val_loss = total_loss_sum / max(total_count, 1)
                val_metric = (total_metric_sum / max(total_count, 1)) if self.metric else 0.0

                # Short summary for validation
                if self.rank == 0:
                    tqdm.write(
                        f"epoch: {epoch + 1}/{self.num_epochs} | "
                        f"val_loss: {val_loss:.4f} | "
                        f"val_metric: {val_metric:.4f}"
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
                self.history['train_loss'].append(avg_loss)
                self.history['train_metric'].append(avg_metric)
                self.history['val_loss'].append(val_loss)
                self.history['val_metric'].append(val_metric)

                # Save a checkpoint every epoch
                self.save_checkpoint(epoch)

                # Log results
                logs = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'train_metric': avg_metric,
                    'val_loss': val_loss,
                    'val_metric': val_metric,
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
        loss_type: str,
        plot: Optional[Union[Callable, List[Callable]]] = None
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        if self.test_loader is None:
            raise ValueError("Test dataset is not provided.")
        
        self.model.eval()
        loss_sum = 0.0
        metric_sum = 0.0
        count = 0
        local_true = []
        local_pred = []

        for X_test, y_test in self.test_loader:
            X_test = X_test.to(self.device, non_blocking=self.pin_memory)
            y_test = y_test.to(self.device, non_blocking=self.pin_memory)

            outputs = self.model(X_test)
            batch_loss = self.criterion(outputs, y_test)
            bsz = y_test.size(0)
            loss_sum += float(batch_loss.item()) * bsz

            if self.metric:
                metric_sum += float(self.metric(outputs, y_test)) * bsz

            count += bsz

            probs = outputs
            if loss_type == 'cross_entropy':
                probs = F.softmax(outputs, dim=1)
            elif loss_type == 'bce':
                probs = torch.sigmoid(outputs)

            local_true.append(y_test.detach().cpu().numpy())
            local_pred.append(probs.detach().cpu().numpy())

        # Stack per-rank arrays
        local_true = np.concatenate(local_true, axis=0) if local_true else np.empty((0,))
        local_pred = np.concatenate(local_pred, axis=0) if local_pred else np.empty((0,))

        # Gather scalars from all processes
        if self._is_distributed:
            packed = torch.tensor(
                data=[loss_sum, metric_sum, float(count)],
                dtype=torch.float64,
                device=self.device,
            )
            gathered = [torch.zeros_like(packed) for _ in range(self.world_size)]
            all_gather(gathered, packed)

            total_loss_sum = sum(g[0].item() for g in gathered)
            total_metric_sum = sum(g[1].item() for g in gathered)
            total_count = int(sum(g[2].item() for g in gathered))
        else:
            total_loss_sum = loss_sum
            total_metric_sum = metric_sum
            total_count = count

        # Global averages
        test_loss = total_loss_sum / max(total_count, 1)
        test_metric = (total_metric_sum / max(total_count, 1)) if self.metric else 0.0

        # Gather variable-length arrays from all processes
        if self._is_distributed:
            gathered_true = [None for _ in range(self.world_size)]
            gathered_pred = [None for _ in range(self.world_size)]
            all_gather_object(gathered_true, local_true)
            all_gather_object(gathered_pred, local_pred)

            if self.rank == 0:
                y_true = np.concatenate(gathered_true, axis=0) if gathered_true else np.empty((0,))
                y_pred = np.concatenate(gathered_pred, axis=0) if gathered_pred else np.empty((0,))
            else:
                y_true = local_true
                y_pred = local_pred
        else:
            y_true = local_true
            y_pred = local_pred

        if self.rank == 0:
            print(f"test_loss: {test_loss:.4f} | test_metric: {test_metric:.4f}")

            # Visualization
            if plot is not None:
                if isinstance(plot, list):
                    for viz in plot:
                        if getattr(viz, '__name__', '') == 'plot_roc_curve':
                            output_path = os.path.join(self.outputs_dir, f"{self.run_name}_roc_curve.png")
                        elif getattr(viz, '__name__', '') == 'plot_confusion_matrix':
                            output_path = os.path.join(self.outputs_dir, f"{self.run_name}_confusion_matrix.png")

                        output_path = output_path if self.save_fig else None
                        viz(y_true, y_pred, save_fig=output_path)
                else:
                    if getattr(plot, '__name__', '') == 'plot_roc_curve':
                        output_path = os.path.join(self.outputs_dir, f"{self.run_name}_roc_curve.png")
                    elif getattr(plot, '__name__', '') == 'plot_confusion_matrix':
                        output_path = os.path.join(self.outputs_dir, f"{self.run_name}_confusion_matrix.png")

                    output_path = output_path if self.save_fig else None
                    plot(y_true, y_pred, save_fig=output_path)

        return test_loss, test_metric, y_true, y_pred