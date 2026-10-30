import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.utils import CFG


class DualHead(nn.Module):
    """
    Shared head for classification and regression.

    Both tasks run off the same backbone features. Classification gives us
    signal vs background, regression predicts a proxy jet mass. Training
    jointly acts as a soft regularizer — in practice it consistently pushes
    AUC up by a small but reliable margin compared to classification alone.
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.cls = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        self.reg = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.cls(x).squeeze(1), self.reg(x).squeeze(1)


class XCiTModel(nn.Module):
    """
    XCiT backbone with dual classification + regression head.

    XCiT replaces the standard token-token attention with cross-covariance
    attention — complexity grows with channel dimension rather than sequence
    length. For dense jet images this matters since we end up with 64 tokens
    at 16x16 patch size and that number scales fast if you go smaller.

    Two initialization modes:
    - Random init (pretrained_imagenet=False): trained from scratch or with
      our own CMS MAE pretraining.
    - ImageNet weights (pretrained_imagenet=True): we load the 3-channel
      pretrained model and adapt the patch embed conv to 8 channels by
      averaging across RGB and repeating. Not ideal but better than
      reinitializing that layer completely.
    """
    def __init__(self, pretrained_imagenet=False):
        super().__init__()

        if pretrained_imagenet:
            bb = timm.create_model(
                CFG['xcit_model'], pretrained=True,
                in_chans=3, img_size=128, num_classes=0
            )

            # timm changed where Conv2d lives inside patch_embed across versions
            # so we search for it rather than hardcoding the path
            conv_layer = None
            for name, module in bb.patch_embed.named_modules():
                if isinstance(module, nn.Conv2d):
                    conv_layer = (name, module)
                    break

            if conv_layer is None:
                raise ValueError("Could not find Conv2d in patch_embed — "
                                 "check timm version")

            layer_name, conv = conv_layer

            # average the 3 RGB filters into 1, then tile to 8 channels
            # this keeps the pretrained spatial structure intact
            w_old = conv.weight.data
            w_new = w_old.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1)

            out_ch, _, kH, kW = w_new.shape
            new_conv = nn.Conv2d(
                8, out_ch, kH,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None
            )
            new_conv.weight.data = w_new
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data.clone()

            # swap the old conv out
            parts  = layer_name.split('.')
            parent = bb.patch_embed
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv)

            self.backbone = bb

        else:
            self.backbone = timm.create_model(
                CFG['xcit_model'], pretrained=False,
                in_chans=8, img_size=128, num_classes=0
            )

        self.head = DualHead(self.backbone.num_features)

    def forward(self, x):
        return self.head(self.backbone(x))

    def forward_features(self, x):
        """Returns raw backbone features — used for MAE pretraining and analysis."""
        return self.backbone(x)


class SwinModel(nn.Module):
    """
    Swin Transformer with dual head — our softmax attention baseline.

    Included mainly to answer the question: does linear attention actually
    help here or is it just the architecture depth? Swin uses shifted window
    attention which is O(N) in practice but still softmax — so any gap
    between Swin and XCiT/L2ViT is attributable to the attention mechanism
    rather than model size or training setup.

    Trained from scratch only — no ImageNet pretraining for a fair comparison
    with the scratch variants of the linear attention models.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG['swin_model'], pretrained=False,
            in_chans=8, img_size=128, num_classes=0
        )
        self.head = DualHead(self.backbone.num_features)

    def forward(self, x):
        return self.head(self.backbone(x))


# ── L2ViT Architecture ────────────────────────────────────────────────────────
#
# L2ViT combines local window attention (LWA) and linear global attention (LGA)
# in each stage. The key insight is that pure linear attention with ReLU tends
# to spread attention too uniformly across tokens — the Local Concentration
# Module (LCM) after each LGA block pulls the focus back toward spatially
# coherent regions, which matters a lot for jet images where the signal is
# concentrated near the jet core.
#
# Stage layout: CPE → LWA → FFN → CPE → LGA → FFN (repeated per block)
# Hierarchy:    stem → stage1 → merge → stage2 → merge → stage3 → merge → stage4


class ConditionalPE(nn.Module):
    """
    Conditional Positional Encoding via depthwise conv.

    Generates position encodings conditioned on the local token content
    rather than using fixed sinusoidal or learned absolute encodings.
    Works well for variable resolution inputs and avoids the interpolation
    issues you get with absolute encodings when the image size changes.
    """
    def __init__(self, channels):
        super().__init__()
        self.pe = nn.Conv2d(channels, channels, kernel_size=3,
                            padding=1, groups=channels)

    def forward(self, tokens, H, W):
        B, N, C = tokens.shape
        spatial = tokens.transpose(1, 2).reshape(B, C, H, W)
        return self.pe(spatial).flatten(2).transpose(1, 2)


class LocalConcentration(nn.Module):
    """
    Local Concentration Module (LCM).

    The core problem with ReLU linear attention is that the output tends to
    be spatially diffuse — every token gets a mix of all other tokens with
    no strong locality bias. LCM fixes this by running two large depthwise
    convs (7x7) after the attention output, which effectively re-introduces
    a local inductive bias without the quadratic cost of window attention.
    """
    def __init__(self, channels, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.conv_in  = nn.Conv2d(channels, channels, kernel,
                                  padding=pad, groups=channels)
        self.activate = nn.GELU()
        self.norm     = nn.BatchNorm2d(channels)
        self.conv_out = nn.Conv2d(channels, channels, kernel,
                                  padding=pad, groups=channels)

    def forward(self, tokens, H, W):
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv_in(x)
        x = self.activate(x)
        x = self.norm(x)
        x = self.conv_out(x)
        return x.flatten(2).transpose(1, 2)


class LinearGlobalAttn(nn.Module):
    """
    Linear Global Attention with ReLU kernel and LCM correction.

    Uses ReLU as the feature map instead of softmax. This keeps Q and K
    non-negative, which is required for the O(N) associativity trick:
    instead of computing (QK^T)V which is O(N^2 d), we compute K^T V first
    (O(N d^2)) and then Q(K^T V) (O(N d^2)). For our 64-token sequences
    this doesn't matter much but it's the principled formulation.

    The denominator clamp at 1e2 is important — without it you get division
    by near-zero values in early training when Q and K are both close to zero
    after ReLU, which causes NaN losses.
    """
    def __init__(self, channels, n_heads=4, lcm_kernel=7):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = channels // n_heads
        self.to_qkv   = nn.Linear(channels, channels * 3)
        self.out_proj = nn.Linear(channels, channels)
        self.pre_norm = nn.LayerNorm(channels)
        self.lcm      = LocalConcentration(channels, lcm_kernel)
        self.lcm_norm = nn.LayerNorm(channels)
        self.scale    = nn.Parameter(
            torch.ones(1) * (self.head_dim ** -0.5)
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        normed  = self.pre_norm(x)

        qkv = self.to_qkv(normed)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.relu(q)
        k = F.relu(k)

        # O(N) formulation: compute K^T V first, then Q(K^T V)
        kv_product = torch.einsum('bhnd,bhnv->bhdv', k, v)
        normalizer = k.sum(dim=2, keepdim=True).clamp(min=1e2)

        attn_out = torch.einsum('bhnd,bhdv->bhnv', q, kv_product)
        denom    = (q * normalizer).sum(-1, keepdim=True).clamp(min=1e2)
        attn_out = attn_out / denom

        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.out_proj(attn_out)

        # LCM sharpens the spatially diffuse linear attention output
        attn_out = attn_out + self.lcm(self.lcm_norm(attn_out), H, W)

        return x + attn_out


class LocalWindowAttn(nn.Module):
    """
    Local Window Attention (LWA).

    Standard softmax attention restricted to non-overlapping 7x7 windows.
    Handles fine-grained local structure that linear global attention misses.
    Images are padded to the nearest multiple of window size before
    partitioning and the padding is cropped back after.
    """
    def __init__(self, channels, n_heads=4, window=7):
        super().__init__()
        self.window   = window
        self.pre_norm = nn.LayerNorm(channels)
        self.attn     = nn.MultiheadAttention(
            channels, n_heads, batch_first=True
        )
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x, H, W):
        B, N, C = x.shape
        ws      = self.window
        normed  = self.pre_norm(x).reshape(B, H, W, C)

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            normed = F.pad(normed, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = normed.shape[1], normed.shape[2]

        windows  = normed.reshape(B, Hp//ws, ws, Wp//ws, ws, C)
        windows  = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws*ws, C)
        attended, _ = self.attn(windows, windows, windows)

        attended = attended.reshape(B, Hp//ws, Wp//ws, ws, ws, C)
        attended = attended.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        if pad_h > 0 or pad_w > 0:
            attended = attended[:, :H, :W, :]

        return x + self.out_proj(attended.reshape(B, N, C))


class FFN(nn.Module):
    """Standard pre-norm feed-forward block with 4x expansion."""
    def __init__(self, channels, expand=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.fc1  = nn.Linear(channels, channels * expand)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(channels * expand, channels)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class L2ViTStage(nn.Module):
    """
    One hierarchical stage of L2ViT.

    Each block runs: CPE → LWA → FFN → CPE → LGA → FFN
    LWA handles local structure, LGA handles global context.
    CPE before each attention block injects spatial position information
    conditioned on the current token content.
    """
    def __init__(self, channels, n_heads, n_blocks, window=7, lcm_kernel=7):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_blocks):
            self.layers.append(nn.ModuleList([
                ConditionalPE(channels),
                LocalWindowAttn(channels, n_heads, window),
                FFN(channels),
                ConditionalPE(channels),
                LinearGlobalAttn(channels, n_heads, lcm_kernel),
                FFN(channels),
            ]))

    def forward(self, tokens, H, W):
        for cpe_lwa, lwa, ffn1, cpe_lga, lga, ffn2 in self.layers:
            tokens = tokens + cpe_lwa(tokens, H, W)
            tokens = lwa(tokens, H, W)
            tokens = ffn1(tokens)
            tokens = tokens + cpe_lga(tokens, H, W)
            tokens = lga(tokens, H, W)
            tokens = ffn2(tokens)
        return tokens


class L2ViTModel(nn.Module):
    """
    L2ViT — hierarchical vision transformer with linear global attention.

    4-stage design inspired by Swin but with LGA replacing shifted window
    attention in the global path. Channel dimensions double at each stage
    (96 → 192 → 384 → 768) and spatial resolution halves via strided convs.

    The convolutional stem (two 3x3 stride-2 convs) gives 4x spatial
    downsampling before stage 1. This is better than a single large patch
    embed for jet images because it builds up local features gradually
    rather than jumping straight to 16x16 patches.

    Architecture:
        stem → stage1(d=1) → merge → stage2(d=1) →
        merge → stage3(d=3) → merge → stage4(d=1) → gap → head
    """
    def __init__(self):
        super().__init__()
        ch     = [96, 192, 384, 768]
        heads  = [3, 6, 12, 24]
        depths = [1, 1, 3, 1]

        self.stem = nn.Sequential(
            nn.Conv2d(8, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, ch[0], kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        self.stage1 = L2ViTStage(ch[0], heads[0], depths[0])
        self.merge1 = nn.Conv2d(ch[0], ch[1], kernel_size=2, stride=2)

        self.stage2 = L2ViTStage(ch[1], heads[1], depths[1])
        self.merge2 = nn.Conv2d(ch[1], ch[2], kernel_size=2, stride=2)

        self.stage3 = L2ViTStage(ch[2], heads[2], depths[2])
        self.merge3 = nn.Conv2d(ch[2], ch[3], kernel_size=2, stride=2)

        self.stage4 = L2ViTStage(ch[3], heads[3], depths[3])

        self.final_norm   = nn.LayerNorm(ch[3])
        self.num_features = ch[3]
        self.head         = DualHead(ch[3])

    def forward_features(self, x):
        """Returns pooled backbone features — used for MAE and analysis."""
        x = self.stem(x)

        for stage, merge in [
            (self.stage1, self.merge1),
            (self.stage2, self.merge2),
            (self.stage3, self.merge3),
        ]:
            B, C, H, W = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            tokens = stage(tokens, H, W)
            x = tokens.transpose(1, 2).reshape(B, C, H, W)
            x = merge(x)

        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.stage4(tokens, H, W)
        tokens = self.final_norm(tokens)
        return tokens.mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_model(name):
    """
    Convenience factory so notebooks don't need to import individual classes.

    Args:
        name: one of 'xcit_mae', 'xcit_scratch', 'xcit_imagenet',
                     'l2vit', 'swin'
    """
    models = {
        'xcit_mae'      : lambda: XCiTModel(pretrained_imagenet=False),
        'xcit_scratch'  : lambda: XCiTModel(pretrained_imagenet=False),
        'xcit_imagenet' : lambda: XCiTModel(pretrained_imagenet=True),
        'l2vit'         : lambda: L2ViTModel(),
        'swin'          : lambda: SwinModel(),
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Choose from: {list(models.keys())}")
    return models[name]()