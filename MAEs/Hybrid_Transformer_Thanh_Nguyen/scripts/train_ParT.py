import os
import yaml
import argparse
import warnings

import torch
import torch.multiprocessing as mp

from src.configs import ParticleTransformerConfig, TrainConfig
from src.engine import JetClassTrainer, MaskedModelTrainer
from src.models import ParticleTransformer
from src.utils import accuracy_metric_ce, set_seed, setup_ddp, cleanup_ddp
from src.utils.data import LazyJetClassDataset
from src.utils.viz import plot_history, plot_ssl_history

warnings.filterwarnings('ignore')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ParticleTransformer from YAML config")

    # Model and configurations arguments
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--config-path', type=str, default='./configs/train_ParT.yaml', help="Path to YAML config")
    parser.add_argument('--checkpoint-path', type=str, default=None, help="Checkpoint to restore trainer state")

    # Data loading arguments
    parser.add_argument('--train-data-dir', type=str, default='./data/train_100M', help="Train data folder")
    parser.add_argument('--val-data-dir', type=str, default='./data/val_5M', help="Validation data folder")

    return parser.parse_args()


def main(
    rank: int,
    world_size: int,
    seed: int,
    config_path: str,
    checkpoint_path: str = None,
    train_data_dir: str = './data/train_100M',
    val_data_dir: str = './data/val_5M'
):
    # Reproducibility settings
    set_seed(seed)
    
    # Load the YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = ParticleTransformerConfig.from_dict(config['model'])
    train_config = TrainConfig.from_dict(config['train'])

    # Initialize multi-GPU processing
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Normalization settings
    normalize = [True, False, False, True]
    norm_dict = {
        'pT': (92.72917175292969, 105.83937072753906),
        'eta': (0.0005733045982196927, 0.9174848794937134),
        'phi': (-0.00041169871110469103, 1.8136887550354004),
        'energy': (133.8745574951172, 167.528564453125)
    }
    
    # Broadcast normalization stats to all processes
    obj_list = [norm_dict]
    torch.distributed.broadcast_object_list(obj_list, src=0)
    norm_dict = obj_list[0]

    # Create the dataset
    if model_config.mask:
        train_dataset = LazyJetClassDataset(train_data_dir, normalize, norm_dict, mask_mode='biased')
        val_dataset = LazyJetClassDataset(val_data_dir, normalize, norm_dict, mask_mode='biased')
    else:
        train_dataset = LazyJetClassDataset(train_data_dir, normalize, norm_dict, mask_mode=None)
        val_dataset = LazyJetClassDataset(val_data_dir, normalize, norm_dict, mask_mode=None)

    # Initialize the model
    model = ParticleTransformer(config=model_config).to(device)

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
        trainer = JetClassTrainer(
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
        output_path = os.path.join(trainer.outputs_dir, f"{trainer.run_name}.png") if train_config.save_fig else None
        if model_config.mask:
            plot_ssl_history(history, save_fig=output_path)
        else:
            plot_history(history, save_fig=output_path)


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    # Multi-GPU processing
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            main,
            args=(
                world_size,
                args.seed,
                args.config_path,
                args.checkpoint_path,
                args.train_data_dir,
                args.val_data_dir
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
            val_data_dir=args.val_data_dir
        )