import os
import random
import numpy as np
import torch

CFG = {
    # paths — update these based on your environment
    'train_path'      : './data/cms_jet_train.h5',
    'test_path'       : './data/cms_jet_test.h5',
    'unlabelled_path' : './data/Dataset_Specific_Unlabelled.h5',

    # output dirs
    'save_dir'        : './weights',
    'results_dir'     : './results',
    'plots_dir'       : './plots',
    'history_dir'     : './history',

    # image
    'img_size'        : 125,
    'img_size_model'  : 128,   # padded to 128 for patch divisibility
    'in_chans'        : 8,
    'pixel_max'       : 255.0,

    # reproducibility
    'seed'            : 42,

    # data split
    'val_size'        : 400,

    # pretraining
    'pretrain_epochs' : 50,
    'pretrain_lr'     : 1.5e-4,
    'pretrain_batch'  : 128,
    'mask_ratio'      : 0.75,
    'pretrain_warmup' : 5,

    # finetuning
    'finetune_batch'  : 128,
    'weight_decay'    : 0.05,
    'grad_clip'       : 1.0,
    'label_smoothing' : 0.1,
    'lambda_reg'      : 0.1,   # regression loss weight
    'use_amp'         : True,

    # stage 1 — heads only
    'stage1_epochs'   : 30,
    'stage1_lr'       : 1e-3,
    'stage1_patience' : 8,

    # stage 2 — full model
    'stage2_epochs'   : 50,
    'stage2_lr_ft'    : 5e-6,  # pretrained backbone
    'stage2_lr_scr'   : 1e-5,  # scratch
    'stage2_patience' : 10,

    # model names
    'xcit_model'      : 'xcit_small_12_p16_224',
    'swin_model'      : 'swin_tiny_patch4_window7_224',
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dirs():
    for key in ['save_dir', 'results_dir', 'plots_dir', 'history_dir']:
        os.makedirs(CFG[key], exist_ok=True)