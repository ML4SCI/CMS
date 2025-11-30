import yaml
import argparse

import torch
import torch.multiprocessing as mp

from src.configs import LorentzParTConfig, TrainConfig
from src.engine import JetClassTrainer, MaskedModelTrainer
from src.models import LorentzParT
from src.utils import accuracy_metric_ce, set_seed, setup_ddp, cleanup_ddp
from src.utils.data import LazyJetClassDataset
from src.utils.viz import plot_particle_reconstruction, plot_confusion_matrix, plot_roc_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LorentzParT from YAML config")

    # Model and configurations arguments
    parser.add_argument('--config-path', type=str, default='./configs/train_LorentzParT.yaml', help="Path to YAML config")
    parser.add_argument('--best-model-path', type=str, default='./logs/LorentzParT/best/run_01.pt', help="Path to best model weights")

    # Data loading arguments
    parser.add_argument('--test-data-dir', type=str, default='./data/test_20M', help="Test data folder")

    return parser.parse_args()


@torch.no_grad()
def main(
    rank: int,
    world_size: int,
    config_path: str,
    best_model_path: str = None,
    test_data_dir: str = './data/test_20M'
):
    # Reproducibility settings
    set_seed(42)

    # Load the YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = LorentzParTConfig.from_dict(config['model'])
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
        test_dataset = LazyJetClassDataset(test_data_dir, normalize, norm_dict, mask_mode='first')
    else:
        test_dataset = LazyJetClassDataset(test_data_dir, normalize, norm_dict, mask_mode=None)

    # Initialize the model
    model = LorentzParT(config=model_config).to(device)

    # Trainer stub for evaluation convenience
    if model_config.mask:
        trainer = MaskedModelTrainer(
            model=model,
            train_dataset=test_dataset,
            val_dataset=test_dataset,
            test_dataset=test_dataset,
            device=device,
            config=train_config
        )
    else:
        trainer = JetClassTrainer(
            model=model,
            train_dataset=test_dataset,
            val_dataset=test_dataset,
            test_dataset=test_dataset,
            device=device,
            metric=accuracy_metric_ce,
            config=train_config
        )

    # Load the best model
    trainer.load_best_model(best_model_path)

    # Evaluate the model
    if model_config.mask:
        test_loss, test_metric, y_true, y_pred = trainer.evaluate(plot_particle_reconstruction)
    else:
        test_loss, test_metric, y_true, y_pred = trainer.evaluate(
            loss_type='cross_entropy',
            plot=[plot_roc_curve, plot_confusion_matrix]
        )

    # Clean up distributed processing
    cleanup_ddp()


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
                args.config_path,
                args.best_model_path,
                args.test_data_dir
            ),
            nprocs=world_size
        )
    else:
        # 1 GPU or CPU: run the same code on rank 0
        main(
            rank=0,
            world_size=1,
            config_path=args.config_path,
            best_model_path=args.best_model_path,
            test_data_dir=args.test_data_dir
        )