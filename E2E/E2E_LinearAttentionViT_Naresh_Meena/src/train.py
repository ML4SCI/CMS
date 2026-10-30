import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score

from src.utils import CFG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def smooth_bce(logits, labels, s=0.1):
    """Label smoothed BCE — reduces overconfident predictions."""
    soft = labels * (1 - s) + 0.5 * s
    return F.binary_cross_entropy_with_logits(logits, soft)


def train_epoch(model, loader, optimizer, scaler, cfg):
    model.train()
    tot = cls_t = reg_t = 0.0
    mse = nn.MSELoss()

    for imgs, labels, mass in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mass   = mass.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=cfg['use_amp']):
            cls_out, reg_out = model(imgs)
            l_cls = smooth_bce(cls_out, labels, cfg['label_smoothing'])
            l_reg = mse(reg_out, mass)
            loss  = l_cls + cfg['lambda_reg'] * l_reg

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        tot   += loss.item()
        cls_t += l_cls.item()
        reg_t += l_reg.item()

    n = len(loader)
    return tot / n, cls_t / n, reg_t / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    tot_loss = 0.0

    for imgs, labels, _ in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=CFG['use_amp']):
            cls_out, _ = model(imgs)

        tot_loss += F.binary_cross_entropy_with_logits(cls_out, labels).item()
        all_logits.append(cls_out.float().cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)

    return {
        'loss'   : tot_loss / len(loader),
        'auc'    : roc_auc_score(labels, probs),
        'acc'    : accuracy_score(labels, preds),
        'f1'     : f1_score(labels, preds),
        'probs'  : probs,
        'labels' : labels,
    }


def run_mae(model, loader, save_path, cfg, model_name='model'):
    """
    MAE pretraining loop.

    Saves the backbone state dict at the best reconstruction loss.
    History is saved as a numpy array for later plotting.
    """
    from src.mae import MAEPretrainer

    backbone = model.backbone if hasattr(model, 'backbone') else model
    mae      = MAEPretrainer(
        backbone,
        patch_size = 16,
        mask_ratio = cfg['mask_ratio'],
        img_size   = cfg['img_size_model'],
        in_chans   = cfg['in_chans']
    ).to(device)

    optimizer = torch.optim.AdamW(
        mae.parameters(),
        lr=cfg['pretrain_lr'],
        weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['pretrain_epochs']
    )
    scaler    = GradScaler(enabled=cfg['use_amp'])
    best_loss = float('inf')
    history   = []

    print(f"\nMAE Pretraining — {model_name}")
    print(f"{'='*50}")

    for ep in range(cfg['pretrain_epochs']):
        mae.train()
        ep_loss, n_batches = 0.0, 0
        t0 = time.time()

        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast(enabled=cfg['use_amp']):
                loss = mae(imgs)

            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(mae.parameters(), cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            ep_loss   += loss.item()
            n_batches += 1

        scheduler.step()
        avg = ep_loss / max(n_batches, 1)
        history.append(avg)

        if avg < best_loss:
            best_loss = avg
            torch.save(backbone.state_dict(), save_path)

        print(f"  ep {ep+1:3d}/{cfg['pretrain_epochs']}  "
              f"loss={avg:.5f}  best={best_loss:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"t={time.time()-t0:.0f}s")

    np.save(
        os.path.join(cfg['history_dir'], f'pretrain_{model_name}.npy'),
        np.array(history)
    )

    # restore best backbone weights into the model
    backbone.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\n  Best loss : {best_loss:.5f}")
    print(f"  Saved     : {save_path}")
    return history


def finetune(model, train_loader, val_loader, cfg, label, use_low_lr=True):
    """
    Two-stage finetuning.

    Stage 1 — freeze backbone, train heads only. Gets the heads to a
    reasonable starting point before we touch the pretrained weights.

    Stage 2 — unfreeze everything at a much lower lr. For pretrained
    models we use 5e-6 to avoid destroying the MAE representations,
    for scratch models 1e-5 since there's nothing to preserve.

    Both stages use early stopping and save the best val AUC checkpoint.
    """
    save_path = os.path.join(cfg['save_dir'], f'{label}_best.pth')
    history   = {k: [] for k in [
        'train_loss', 'train_cls', 'train_reg',
        'val_loss', 'val_auc', 'val_acc', 'val_f1', 'stage'
    ]}
    best_auc = 0.0
    best_ep  = 0
    g_ep     = 0

    # ── Stage 1 ───────────────────────────────────────────────
    print(f"\nFinetuning — {label}")
    print(f"{'='*50}")
    print("\nStage 1 — backbone frozen")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    print(f"  Trainable params : "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['stage1_lr'],
        weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['stage1_epochs']
    )
    scaler = GradScaler(enabled=cfg['use_amp'])
    no_imp = 0

    for ep in range(cfg['stage1_epochs']):
        t0 = time.time()
        tl, tc, tr = train_epoch(model, train_loader, optimizer, scaler, cfg)
        scheduler.step()
        vm   = evaluate(model, val_loader)
        g_ep += 1

        for k, v in zip(history.keys(),
                        [tl, tc, tr, vm['loss'], vm['auc'],
                         vm['acc'], vm['f1'], 1]):
            history[k].append(v)

        tag = ''
        if vm['auc'] > best_auc:
            best_auc = vm['auc']
            best_ep  = g_ep
            no_imp   = 0
            torch.save(model.state_dict(), save_path)
            tag = ' ✓'
        else:
            no_imp += 1

        print(f"  S1 ep {ep+1:2d}/{cfg['stage1_epochs']}  "
              f"loss={tl:.4f}  cls={tc:.4f}  reg={tr:.5f}  "
              f"val_auc={vm['auc']:.4f}  val_acc={vm['acc']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"t={time.time()-t0:.0f}s{tag}")

        if no_imp >= cfg['stage1_patience']:
            print(f"  Early stop at ep {ep+1}")
            break

    print(f"\n  Stage 1 best AUC : {best_auc:.4f} (ep {best_ep})")
    model.load_state_dict(torch.load(save_path, map_location=device))

    # ── Stage 2 ───────────────────────────────────────────────
    print("\nStage 2 — full model")

    for p in model.parameters():
        p.requires_grad = True

    print(f"  Trainable params : "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    lr2       = cfg['stage2_lr_ft'] if use_low_lr else cfg['stage2_lr_scr']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr2,
        weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['stage2_epochs']
    )
    scaler = GradScaler(enabled=cfg['use_amp'])
    no_imp = 0

    for ep in range(cfg['stage2_epochs']):
        t0 = time.time()
        tl, tc, tr = train_epoch(model, train_loader, optimizer, scaler, cfg)
        scheduler.step()
        vm   = evaluate(model, val_loader)
        g_ep += 1

        for k, v in zip(history.keys(),
                        [tl, tc, tr, vm['loss'], vm['auc'],
                         vm['acc'], vm['f1'], 2]):
            history[k].append(v)

        tag = ''
        if vm['auc'] > best_auc:
            best_auc = vm['auc']
            best_ep  = g_ep
            no_imp   = 0
            torch.save(model.state_dict(), save_path)
            tag = ' ✓'
        else:
            no_imp += 1

        print(f"  S2 ep {ep+1:2d}/{cfg['stage2_epochs']}  "
              f"loss={tl:.4f}  cls={tc:.4f}  reg={tr:.5f}  "
              f"val_auc={vm['auc']:.4f}  val_acc={vm['acc']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"t={time.time()-t0:.0f}s{tag}")

        if no_imp >= cfg['stage2_patience']:
            print(f"  Early stop at ep {ep+1}")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    np.save(
        os.path.join(cfg['history_dir'], f'{label}_history.npy'),
        {k: np.array(v) for k, v in history.items()}
    )

    print(f"\n  Best AUC : {best_auc:.4f} (ep {best_ep})")
    print(f"  Saved    : {save_path}")
    return history, best_auc


def final_eval(model, test_loader, label):
    """Evaluate on test set and return a clean results dict."""
    m = evaluate(model, test_loader)

    fpr, tpr, _ = roc_curve(m['labels'], m['probs'])

    def rej_at(eff):
        idx = np.where(tpr >= eff)[0]
        return float(1 - fpr[idx[0]]) if len(idx) else 0.0

    result = {
        'label'  : label,
        'auc'    : round(float(m['auc']), 4),
        'acc'    : round(float(m['acc']), 4),
        'f1'     : round(float(m['f1']),  4),
        'rej_90' : round(rej_at(0.90), 4),
        'rej_95' : round(rej_at(0.95), 4),
        'rej_99' : round(rej_at(0.99), 4),
        'probs'  : m['probs'].tolist(),
        'labels' : m['labels'].tolist(),
    }

    print(f"\n  {label}")
    print(f"  AUC={result['auc']}  Acc={result['acc']}  F1={result['f1']}  "
          f"Rej@90={result['rej_90']}  Rej@95={result['rej_95']}")
    return result


def save_result(name, result, results_file):
    """Append result to the shared JSON file."""
    existing = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            existing = json.load(f)
    existing[name] = result
    with open(results_file, 'w') as f:
        json.dump(existing, f, indent=2)