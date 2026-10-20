import os
import yaml
import argparse
import warnings

import torch
import torch.multiprocessing as mp

from src.utils import check_lgatr_version
check_lgatr_version()  # fail fast with the install command if lgatr is missing or not 1.4.4

from src.configs import ParticleTransformerConfig, TrainConfig
from src.engine import JetClassTrainer, MaskedModelTrainer, Trainer
from src.models import ParticleTransformer
from src.utils import accuracy_metric_ce, set_seed, setup_ddp, cleanup_ddp
from src.utils.data import JetClassDataset, LazyJetClassDataset, compute_norm_stats, load_npz_splits
from src.utils.viz import plot_history, plot_ssl_history

warnings.filterwarnings('ignore')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ParticleTransformer from YAML config")

    # Model and configurations arguments
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--config-path', type=str, default='./configs/train_PAG_ParT.yaml', help="Path to YAML config")
    parser.add_argument('--checkpoint-path', type=str, default=None, help="Checkpoint to restore trainer state")
    parser.add_argument('--weights', type=str, default=None, help="Pretrained weights to fine-tune from (overrides model.weights in the YAML)")

    # Data loading arguments
    parser.add_argument('--train-data-dir', type=str, default='./data/train_100M', help="Train data folder")
    parser.add_argument('--val-data-dir', type=str, default='./data/val_5M', help="Validation data folder")
    parser.add_argument('--npz-path', type=str, default=None, help="Pre-serialised .npz dataset, split 80/10/10 as in the PAG notebook (single GPU); replaces the data folders")

    return parser.parse_args()


def main(
    rank: int,
    world_size: int,
    seed: int,
    config_path: str,
    checkpoint_path: str = None,
    train_data_dir: str = './data/train_100M',
    val_data_dir: str = './data/val_5M',
    weights: str = None,
    npz_path: str = None
):
    # Reproducibility settings
    set_seed(seed)

    # Load the YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = ParticleTransformerConfig.from_dict(config['model'])
    train_config = TrainConfig.from_dict(config['train'])
    if weights is not None:
        model_config.weights = weights

    # Initialize multi-GPU processing
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Normalization settings
    normalize = [True, False, False, True]
    mask_mode = 'biased' if model_config.mask else None

    # Create the dataset
    if npz_path is not None:
        # In-memory data as in the PAG notebook, with statistics from the train split
        splits = load_npz_splits(npz_path)
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        norm_dict = compute_norm_stats(X_train)

        train_dataset = JetClassDataset(X_train, y_train, normalize, norm_dict, mask_mode=mask_mode)
        val_dataset = JetClassDataset(X_val, y_val, normalize, norm_dict, mask_mode=mask_mode)
    else:
        norm_dict = {
            'pT': (92.72917175292969, 105.83937072753906),
            'eta': (0.0005733045982196927, 0.9174848794937134),
            'phi': (-0.00041169871110469103, 1.8136887550354004),
            'energy': (133.8745574951172, 167.528564453125)
        }

        # Broadcast normalization stats to all processes
        if torch.distributed.is_initialized():
            obj_list = [norm_dict]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            norm_dict = obj_list[0]

        train_dataset = LazyJetClassDataset(train_data_dir, normalize, norm_dict, mask_mode=mask_mode)
        val_dataset = LazyJetClassDataset(val_data_dir, normalize, norm_dict, mask_mode=mask_mode)

    # Initialize the model; norm_stats lets the physics-aware gate see the jet mass in GeV
    model = ParticleTransformer(
        config=model_config,
        norm_stats=norm_dict,
        normalize=normalize,
        debug=(rank == 0)
    ).to(device)

    # Initialize the trainer
    if model_config.mask:
        trainer = MaskedModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            config=train_config
        )
    else:
        # The notebook's Trainer for the in-memory data, the ParT-paper JetClassTrainer for the ROOT files
        trainer_cls = Trainer if npz_path is not None else JetClassTrainer
        trainer = trainer_cls(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            metric=accuracy_metric_ce,
            config=train_config
        )

    # Resume checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            trainer.load_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Train the model
    history, model = trainer.train()

    # Clean up distributed processing
    cleanup_ddp()

    # Save the training history plot
    if rank == 0:
        if trainer.best_model_path:
            print(f"Best model saved to: {trainer.best_model_path}")

        history_name = 'pretrain_history' if model_config.mask else 'train_history'
        output_path = os.path.join(trainer.outputs_dir, f"{trainer.run_name}_{history_name}.png") if train_config.save_fig else None
        if model_config.mask:
            plot_ssl_history(history, save_fig=output_path)
        else:
            plot_history(history, save_fig=output_path)

        if output_path:
            print(f"Training history plot saved to: {output_path}")


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # The in-memory .npz datasets have no per-file layout for the distributed sampler, so they run on one GPU
    world_size = 1 if args.npz_path else torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            main,
            args=(
                world_size,
                args.seed,
                args.config_path,
                args.checkpoint_path,
                args.train_data_dir,
                args.val_data_dir,
                args.weights,
                args.npz_path
            ),
            nprocs=world_size
        )
    else:
        # 1 GPU or CPU: run the same code on rank 0
        main(
            rank=0,
            world_size=1,
            seed=args.seed,
            config_path=args.config_path,
            checkpoint_path=args.checkpoint_path,
            train_data_dir=args.train_data_dir,
            val_data_dir=args.val_data_dir,
            weights=args.weights,
            npz_path=args.npz_path
        )
