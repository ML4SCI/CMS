import math
from collections import OrderedDict
from functools import partial
from typing import Callable, NamedTuple, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from torchvision.ops.misc import MLP
from torchvision.utils import _log_api_usage_once


class CrossDepthMultiheadSelfAttention(nn.Module):
    def __init__(self, k_factor, in_channels, k_chunk_size: int = 0):
        super(CrossDepthMultiheadSelfAttention, self).__init__()
        self.k_factor = k_factor
        self.in_channels = in_channels
                                                           
                                                                               
        self.k_chunk_size = k_chunk_size if k_chunk_size > 0 else k_factor

        self.qkv_weight = nn.Parameter(torch.randn(in_channels, k_factor, k_factor * 3))
        self.qkv_bias = nn.Parameter(torch.randn(in_channels, k_factor * 3))
        self.fc_out = nn.Linear(k_factor, k_factor)

        self.reset_parameters()

    def reset_parameters(self):
                                                                                    
        for i in range(self.in_channels):
            nn.init.kaiming_uniform_(self.qkv_weight[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.qkv_weight[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.qkv_bias[i], -bound, bound)

        self.fc_out.reset_parameters()

    def _attend_chunk(self, q, k, v):
                                                                   
        queries = q.transpose(-1, -2).unsqueeze(-1)
        keys = k.transpose(-1, -2).unsqueeze(-2)
                                  
        scores = torch.matmul(queries, keys) / (self.k_factor ** 0.5)
        attention = F.softmax(scores, -1)
                                                                                        
        context = torch.einsum('ijklm,ijlk->ijkm', attention, v)
                          
        return context.transpose(-2, -1)

    def forward(self, x):
        batch_size, length, _ = x.shape
                                                                                   
        x_channels = x.view(batch_size, length, self.in_channels, self.k_factor)

                                                                                                        
        qkv = torch.einsum('blck, ckm -> blcm', x_channels, self.qkv_weight)
        qkv = qkv + self.qkv_bias.unsqueeze(0).unsqueeze(0)                                                 

                                                                       
        q, k, v = qkv.chunk(3, dim=-1)                     

        K = self.k_factor
        cs = self.k_chunk_size

        if cs >= K:
                                               
            context = self._attend_chunk(q, k, v)
        else:
                                                                             
            chunks = []
            for start in range(0, K, cs):
                end = min(start + cs, K)
                chunks.append(self._attend_chunk(
                    q[:, :, :, start:end],
                    k[:, :, :, start:end],
                    v[:, :, :, start:end],
                ))
            context = torch.cat(chunks, dim=-1)                

        out = self.fc_out(context)

                                                                  
        out = out.flatten(start_dim=2)

        return out


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
                                                                                                 
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_channels: int,
        k_factor: int,
        mlp_dim: int,
        dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        k_chunk_size: int = 0,
    ):
        super().__init__()

                         
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = CrossDepthMultiheadSelfAttention(k_factor, in_channels, k_chunk_size)
        self.dropout = nn.Dropout(dropout)

                   
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor):
        torch._assert(x.dim() == 3, f"Expected (batch_size, num_particles, hidden_dim) got {x.shape}")
        x = x + self.dropout(self.self_attention(self.ln_1(x)))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_channels: int,
        k_factor: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        max_image_size: int,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        num_hap_layers: int = 0,
        hap_window_size: int = 8,
        hap_mlp_ratio: float = 4.0,
        hap_drop_path: float = 0.0,
        grad_checkpointing: bool = False,
        k_chunk_size: int = 0,
        compile_blocks: bool = False,
        hap_alpha: float = 1.0,
        hap_learnable_alpha: bool = False,
        pos_embed_2d: bool = False,
    ):
        super().__init__()
        self.pos_embed_2d = pos_embed_2d
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.grad_checkpointing = grad_checkpointing

        blocks: List[nn.Module] = []
        is_hap: List[bool] = []

        if num_hap_layers > 0 and num_layers > 0:
            dp_rates = torch.linspace(0, hap_drop_path, num_hap_layers).tolist()
            group_size = num_layers // num_hap_layers
            remainder = num_layers % num_hap_layers
            hap_idx = 0
            for g in range(num_hap_layers):
                n_enc = group_size + (1 if g < remainder else 0)
                for _ in range(n_enc):
                    blocks.append(EncoderBlock(hidden_dim, num_channels, k_factor,
                                               mlp_dim, dropout, norm_layer,
                                               k_chunk_size=k_chunk_size))
                    is_hap.append(False)
                blocks.append(HAPBlock(dim=hidden_dim, window_size=hap_window_size,
                                       mlp_ratio=hap_mlp_ratio, drop=dropout,
                                       drop_path=dp_rates[hap_idx]))
                is_hap.append(True)
                hap_idx += 1
        else:
            for _ in range(num_layers):
                blocks.append(EncoderBlock(hidden_dim, num_channels, k_factor,
                                           mlp_dim, dropout, norm_layer,
                                           k_chunk_size=k_chunk_size))
                is_hap.append(False)

                                                                 
                                                                  
                                                                      
                                                                     
                                                                       
                                            
        if compile_blocks and hasattr(torch, "compile"):
            if grad_checkpointing:
                blocks = [torch.compile(b) if not is_hap[i] else b
                          for i, b in enumerate(blocks)]
            else:
                blocks = [torch.compile(b) for b in blocks]

        self.blocks = nn.ModuleList(blocks)
        self._is_hap = is_hap                                      
        self.has_hap = num_hap_layers > 0

                                                  
                                                                          
                                                                  
                                                                                
                                                                               
                         
        n_hap = max(num_hap_layers, 1)
        if hap_learnable_alpha:
            self.hap_alphas = nn.Parameter(torch.full((n_hap,), float(hap_alpha)))
        else:
            self.register_buffer('hap_alphas', torch.full((n_hap,), float(hap_alpha)))
        self._hap_learnable_alpha = hap_learnable_alpha

        self.ln = norm_layer(hidden_dim)
        self.max_image_size = list(max_image_size)
        self.enc_pos_embedding = nn.Parameter(
            torch.empty(1, max_image_size[0] * max_image_size[1], hidden_dim).normal_(std=0.02)
        )

    def _make_pos_idx(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        max_L = self.enc_pos_embedding.shape[1]
        if length > max_L:
            raise ValueError(f"length={length} exceeds max length {max_L}")
        starts = torch.randint(0, max_L - length + 1, (batch_size,), device=device, dtype=torch.long)
        offsets = torch.arange(length, device=device, dtype=torch.long)
        return starts[:, None] + offsets[None, :]

    def _make_pos_idx_2d(self, batch_size: int, n_h: int, n_w: int,
                         device: torch.device) -> torch.Tensor:
        """Row/col indices into the MAX grid, so token (r, c) maps to the same
        embedding slot regardless of the current image's n_w. Required for
        variable-resolution / bucketed inputs; the 1D version below assigns
        slot r*n_w+c, which silently changes meaning when n_w changes."""
        max_h, max_w = self.max_image_size
        if n_h > max_h or n_w > max_w:
            raise ValueError(f"grid {n_h}x{n_w} exceeds max {max_h}x{max_w}")
        r0 = torch.randint(0, max_h - n_h + 1, (batch_size,), device=device, dtype=torch.long)
        c0 = torch.randint(0, max_w - n_w + 1, (batch_size,), device=device, dtype=torch.long)
        r = torch.arange(n_h, device=device, dtype=torch.long)
        c = torch.arange(n_w, device=device, dtype=torch.long)
        idx = (r0[:, None, None] + r[None, :, None]) * max_w\
            + (c0[:, None, None] + c[None, None, :])
        return idx.reshape(batch_size, n_h * n_w)

    def masking_from_bool_mask(self, x: torch.Tensor, mask: torch.Tensor):
        N, L, D = x.shape
        if mask.dtype is not torch.bool:
            mask = mask > 0
        if mask.shape != (N, L):
            raise ValueError(f"mask must be shape {(N, L)}; got {tuple(mask.shape)}")

        keep = ~mask
        num_keep = keep.sum(dim=1)
        if not torch.all(num_keep == num_keep[0]):
            raise ValueError("All samples must keep the same number of tokens")
        len_keep = int(num_keep[0].item())

        ids = torch.arange(L, device=x.device).unsqueeze(0).expand(N, -1)
        keys = mask.to(torch.long) * L + ids                                       
        ids_shuffle = torch.argsort(keys, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(N, len_keep, D))

                        
        mask_out = mask.to(dtype=torch.float32, device=x.device)
        return x_masked, mask_out, ids_restore

    def forward(
        self,
        input: torch.Tensor,
        n_h: int,
        n_w: int,
        mask: torch.Tensor,
        pos_idx: Optional[torch.Tensor] = None,
    ):
        B, L, D = input.shape

        if pos_idx is None:
            if self.pos_embed_2d:
                pos_idx = self._make_pos_idx_2d(B, n_h, n_w, device=input.device)
            else:
                pos_idx = self._make_pos_idx(B, L, device=input.device)
        enc_pos = self.enc_pos_embedding[0, pos_idx, :]

        x = self.dropout(input + enc_pos)

                                                                     
                                                       
                                                             
                                                                 
        use_ckpt = self.grad_checkpointing and torch.is_grad_enabled()
        hap_idx = 0
        for i, block in enumerate(self.blocks):
            if self._is_hap[i]:
                x_2d = x.view(B, n_h, n_w, D)
                x_hap = grad_checkpoint(block, x_2d, use_reentrant=False) if use_ckpt else block(x_2d)
                alpha = self.hap_alphas[hap_idx]
                x = (x_2d + alpha * (x_hap - x_2d)).reshape(B, n_h * n_w, D)
                hap_idx += 1
            else:
                x = grad_checkpoint(block, x, use_reentrant=False) if use_ckpt else block(x)

                                                                         
        x, mask_out, ids_restore = self.masking_from_bool_mask(x, mask)
        return self.ln(x), mask_out, ids_restore


class TokenAttnHead(nn.Module):
    def __init__(self, k_factor: int, num_classes: int, rank: int = 16, m_factor: int = 1):
        super().__init__()
        self.k = k_factor
        self.r = rank
        self.m = m_factor                                  

                                              
        self.q_depth = nn.Parameter(torch.empty(self.m, self.k).normal_(std=0.02))          

                                                        
        self.q_tokens = nn.Parameter(torch.empty(self.r, self.m * self.k).normal_(std=0.02))           

                                       
        self.ln = nn.LayerNorm(self.m * self.k)

                                                                               
        self.fc = nn.Linear(self.r * self.m * self.k, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L, D = tokens.shape
        C = D // self.k
        x = tokens.view(B, L, C, self.k)                

                                                         
                                                      
        s_c = torch.einsum('mk,blck->blcm', self.q_depth, x) / (self.k ** 0.5)

                                                                
        w_c = s_c.softmax(dim=2)                

                                                        
        s_mk = torch.einsum('blcm,blck->blmk', w_c, x)                

                                                                         
        s = s_mk.reshape(B, L, self.m * self.k)

                                                                     
        s_l = torch.einsum('rf,blf->blr', self.q_tokens, s) / ((self.m * self.k) ** 0.5)             
        a = s_l.softmax(dim=1)                         

        pooled = torch.einsum('blr,blf->brf', a, s)              

        z = self.ln(pooled).reshape(B, -1)             
        return self.fc(z)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    return x, (pad_h, pad_w)


def make_power2_shifts(max_step: int, include_negative: bool = True) -> List[Tuple[int, int]]:
    steps = []
    s = 1
    while s <= max_step:
        steps.append(s)
        s *= 2

    shifts: List[Tuple[int, int]] = []
    for step in steps:
        shifts.append((step, 0))
        shifts.append((0, step))
        if include_negative:
            shifts.append((-step, 0))
            shifts.append((0, -step))

    seen = set()
    uniq = []
    for sh in shifts:
        if sh != (0, 0) and sh not in seen:
            uniq.append(sh)
            seen.add(sh)
    return uniq


class PermuteMixer2D(nn.Module):
    def __init__(self, dim: int, shifts: List[Tuple[int, int]], init_scale: float = 0.0):
        super().__init__()
        self.shifts = list(shifts)
        S = len(self.shifts)
        if init_scale == 0.0:
            self.alpha = nn.Parameter(torch.zeros(S, dim))
        else:
            self.alpha = nn.Parameter(init_scale * torch.randn(S, dim))

    def forward(self, x: torch.Tensor, dims: Tuple[int, int]) -> torch.Tensor:
        out = torch.zeros_like(x)
        view_shape = [1] * (x.ndim - 1) + [x.shape[-1]]
        for i, (dy, dx) in enumerate(self.shifts):
            w = self.alpha[i].view(*view_shape)
            out = out + torch.roll(x, shifts=(dy, dx), dims=dims) * w
        return out


class WindowSelfAttention2D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: Optional[int] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos_bias: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.ws = int(window_size)
        self.num_heads = int(num_heads) if num_heads is not None else max(1, self.dim // 32)
        assert self.dim % self.num_heads == 0, f"dim={dim} must be divisible by num_heads={self.num_heads}"
        self.head_dim = self.dim // self.num_heads

        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_rel_pos_bias:
            num_rel = (2 * self.ws - 1) * (2 * self.ws - 1)
            self.rel_pos_bias = nn.Parameter(torch.zeros(num_rel, self.num_heads))

            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(self.ws, dtype=torch.long),
                    torch.arange(self.ws, dtype=torch.long),
                    indexing="ij",
                )
            )
            coords_flat = coords.reshape(2, -1)
            rel = coords_flat[:, :, None] - coords_flat[:, None, :]
            rel = rel.permute(1, 2, 0).contiguous()
            rel[:, :, 0] += self.ws - 1
            rel[:, :, 1] += self.ws - 1
            rel[:, :, 0] *= 2 * self.ws - 1
            rel_index = rel.sum(-1)
            self.register_buffer("rel_index", rel_index, persistent=False)
        else:
            self.rel_pos_bias = None
            self.rel_index = None

    def forward(self, xw: torch.Tensor) -> torch.Tensor:
        B, Wh, Ww, ws1, ws2, C = xw.shape
        assert ws1 == self.ws and ws2 == self.ws, f"expected ws={self.ws}, got {ws1}x{ws2}"

        N = self.ws * self.ws
        x = xw.reshape(B * Wh * Ww, N, C)
        Bn = x.shape[0]

        qkv = (
            self.qkv(x)
            .reshape(Bn, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_bias = None
        if self.rel_pos_bias is not None:
            bias = self.rel_pos_bias[self.rel_index.reshape(-1)]
            bias = bias.reshape(N, N, self.num_heads).permute(2, 0, 1)
            attn_bias = bias.unsqueeze(0)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            scale = self.head_dim ** -0.5
            attn = (q * scale) @ k.transpose(-2, -1)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(Bn, N, C)
        out = self.proj_drop(self.proj(out))
        return out.reshape(B, Wh, Ww, ws1, ws2, C)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class HAPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        micro_shifts: Optional[List[Tuple[int, int]]] = None,
        macro_shifts: Optional[List[Tuple[int, int]]] = None,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)

        if micro_shifts is None:
            micro_shifts = make_power2_shifts(max_step=max(1, self.window_size // 2), include_negative=True)
        if macro_shifts is None:
            macro_shifts = make_power2_shifts(max_step=4, include_negative=True)

        self.micro_mix = PermuteMixer2D(dim, micro_shifts, init_scale=0.0)
        self.macro_mix = PermuteMixer2D(dim, macro_shifts, init_scale=0.0)

        self.cross1 = nn.Linear(dim, dim)
        self.cross2 = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio), drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _hier_delta(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        ws = self.window_size
        Wh = H // ws
        Ww = W // ws

        xw = (
            x.view(B, Wh, ws, Ww, ws, C)
             .permute(0, 1, 3, 2, 4, 5)
             .contiguous()
        )

        delta = torch.zeros_like(xw)

        d_micro = self.micro_mix(xw, dims=(3, 4))
        delta = delta + d_micro
        xw = xw + d_micro

        macro = xw.mean(dim=(3, 4))
        d_cross1 = self.cross1(macro).unsqueeze(3).unsqueeze(4)
        delta = delta + d_cross1
        xw = xw + d_cross1

        macro = macro + self.macro_mix(macro, dims=(1, 2))

        d_cross2 = self.cross2(macro).unsqueeze(3).unsqueeze(4)
        delta = delta + d_cross2

        delta = (
            delta.permute(0, 1, 3, 2, 4, 5)
                 .contiguous()
                 .view(B, H, W, C)
        )
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_norm = self.norm1(x)

        x_pad, (pad_h, pad_w) = pad_to_multiple(x_norm, self.window_size)
        delta = self._hier_delta(x_pad)
        if pad_h or pad_w:
            delta = delta[:, :x.shape[1], :x.shape[2], :]

        x = shortcut + self.drop_path(delta)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DepthViT(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        k_factor: int,
        num_layers: int,
        mlp_dim: int,
        linear_rank: int,
        max_image_height: int,
        max_image_width: int,
        dropout: float = 0.0,
        num_classes: int = 1,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        num_hap_layers: int = 0,
        hap_window_size: int = 8,
        hap_mlp_ratio: float = 4.0,
        hap_drop_path: float = 0.0,
        grad_checkpointing: bool = True,
        k_chunk_size: int = 0,
        compile_blocks: bool = False,
        hap_alpha: float = 1.0,
        hap_learnable_alpha: bool = False,
        pos_embed_2d: bool = False,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.patch_size = patch_size
        if type(patch_size) == int:
            self.patch_height = patch_size
            self.patch_width = patch_size
        else:
            self.patch_height = patch_size[0]
            self.patch_width = patch_size[1]
        self.in_channels = in_channels
        self.k_factor = k_factor
        self.hidden_dim = in_channels * k_factor
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.linear_rank = linear_rank
        self.max_image_size = [max_image_height // self.patch_height, max_image_width // self.patch_width]

        self.conv_proj = nn.Conv2d(
            in_channels=in_channels, groups=in_channels, out_channels=in_channels*k_factor, kernel_size=patch_size, stride=patch_size
        )

        self.encoder = Encoder(
            num_layers,
            in_channels,
            k_factor,
            self.hidden_dim,
            mlp_dim,
            dropout,
            self.max_image_size,
            norm_layer,
            num_hap_layers=num_hap_layers,
            hap_window_size=hap_window_size,
            hap_mlp_ratio=hap_mlp_ratio,
            hap_drop_path=hap_drop_path,
            grad_checkpointing=grad_checkpointing,
            k_chunk_size=k_chunk_size,
            compile_blocks=compile_blocks,
            hap_alpha=hap_alpha,
            hap_learnable_alpha=hap_learnable_alpha,
            pos_embed_2d=pos_embed_2d,
        )

        self.token_attn_head = TokenAttnHead(k_factor, num_classes, rank=linear_rank)

                                                                          
                                                                               
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        if isinstance(self.conv_proj, nn.Conv2d):
                                    
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
                                                     
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W  = x.shape
        n_h = H // self.patch_height
        n_w = W // self.patch_width

                                                   
        x = self.conv_proj(x)
                                                                                           
        x = x.reshape(N, C * self.k_factor, n_h * n_w)

                                                                                              
        x = x.permute(0, 2, 1)

        return x

    def forward_cls(self, imgs: torch.Tensor):
        x = self._process_input(imgs)
        N, C, H, W = imgs.shape
        n_h = H // self.patch_height
        n_w = W // self.patch_width
        keep_all = torch.zeros(N, n_h * n_w, device=x.device, dtype=torch.bool)
        enc_output, _, _ = self.encoder(x, n_h, n_w, mask=keep_all)
        return self.token_attn_head(enc_output)

    def forward_features(
        self,
        imgs: torch.Tensor,
        mask: torch.Tensor,
        pos_idx: Optional[torch.Tensor] = None,
    ):
        x = self._process_input(imgs)
        _, _, H, W = imgs.shape
        n_h = H // self.patch_height
        n_w = W // self.patch_width

                                                                            
                                                                                 
                                                                             
                                                                               
                                              
        if mask is not None and mask.any():
            mask_f = mask.unsqueeze(-1).to(x.dtype)              
            x = x * (1.0 - mask_f) + self.mask_token * mask_f

        return self.encoder(x, n_h, n_w, mask=mask, pos_idx=pos_idx)


class IJEPAPredictor(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_channels: int,
        k_factor: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        max_image_size: list[int],
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        num_hap_layers: int = 0,
        hap_window_size: int = 7,
        hap_mlp_ratio: float = 4.0,
        hap_drop_path: float = 0.0,
    ):
        super().__init__()

        blocks: List[nn.Module] = []
        is_hap: List[bool] = []

        if num_hap_layers > 0 and num_layers > 0:
            dp_rates = torch.linspace(0, hap_drop_path, num_hap_layers).tolist()
            group_size = num_layers // num_hap_layers
            remainder = num_layers % num_hap_layers
            for g in range(num_hap_layers):
                n_enc = group_size + (1 if g < remainder else 0)
                for _ in range(n_enc):
                    blocks.append(EncoderBlock(hidden_dim, num_channels, k_factor,
                                               mlp_dim, dropout, norm_layer))
                    is_hap.append(False)
                blocks.append(HAPBlock(dim=hidden_dim, window_size=hap_window_size,
                                       mlp_ratio=hap_mlp_ratio, drop=dropout,
                                       drop_path=dp_rates[g]))
                is_hap.append(True)
        else:
            for _ in range(num_layers):
                blocks.append(EncoderBlock(hidden_dim, num_channels, k_factor,
                                           mlp_dim, dropout, norm_layer))
                is_hap.append(False)

        self.blocks = nn.ModuleList(blocks)
        self._is_hap = is_hap

                                                                                                   
                                                                                           
                                                                                    
        for block in self.blocks:
            if isinstance(block, HAPBlock):
                nn.init.normal_(block.micro_mix.alpha, std=0.01)
                nn.init.normal_(block.macro_mix.alpha, std=0.01)

        self.ln = norm_layer(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(
            torch.empty(1, max_image_size[0] * max_image_size[1], hidden_dim).normal_(std=0.02)
        )

    def forward(self, context_tokens: torch.Tensor, ids_restore: torch.Tensor,
                pos_idx: torch.Tensor, n_h: int, n_w: int):
        B, len_keep, D = context_tokens.shape
        L = ids_restore.shape[1]
        if pos_idx.shape != (B, L):
            raise ValueError(f"pos_idx must be shape {(B, L)}; got {tuple(pos_idx.shape)}")

        mask_tokens = self.mask_token.repeat(B, L - len_keep, 1)
        x_ = torch.cat([context_tokens, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(B, L, D))

        pos = self.pos_embedding[0, pos_idx, :]
        x = self.dropout(x + pos)

        for i, block in enumerate(self.blocks):
            if self._is_hap[i]:
                x = x.view(B, n_h, n_w, D)
                x = block(x)
                x = x.reshape(B, L, D)
            else:
                x = block(x)

            if hasattr(self, "rev_op") and hasattr(src, "rev_op"):
                W_t, W_s = self.rev_op.conv_transpose.weight, src.rev_op.conv_transpose.weight
                b_t, b_s = self.rev_op.conv_transpose.bias, src.rev_op.conv_transpose.bias

                                                                                     
                for s_in, t_in, sblk_in, tblk_in in pairs():
                    for s_out, t_out, _, _ in pairs():
                                                        
                        W_t[tblk_in, t_out].copy_(W_s[sblk_in, s_out])

                if b_t is not None and b_s is not None:
                    for s_out, t_out, _, _ in pairs():
                        b_t[t_out].copy_(b_s[s_out])

            if hasattr(self, "token_attn_head") and hasattr(src, "token_attn_head"):
                for p_t, p_s in zip(self.token_attn_head.parameters(), src.token_attn_head.parameters()):
                    p_t.copy_(p_s)
        return self.ln(x)
