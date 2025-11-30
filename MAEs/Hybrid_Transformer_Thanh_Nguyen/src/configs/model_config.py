from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class BaseModelConfig:
    num_classes: int = 10
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 8
    num_cls_layers: int = 2
    num_mlp_layers: int = 0
    hidden_dim: int = 256
    dropout: float = 0.1
    max_num_particles: int = 128
    num_particle_features: int = 4
    expansion_factor: int = 4
    mask: bool = False
    weights: Optional[str] = None
    inference: bool = False


@dataclass
class ParticleTransformerConfig(BaseModelConfig):
    pair_embed_dims: List[int] = field(default_factory=lambda: [64, 64, 64])

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)
    

@dataclass
class LGATrConfig(BaseModelConfig):
    hidden_mv_channels: int = 8
    in_s_channels: int = None
    out_s_channels: int = None
    hidden_s_channels: int = 16
    attention: Optional[Dict] = None
    mlp: Optional[Dict] = None
    reinsert_mv_channels: Optional[Tuple[int]] = None
    reinsert_s_channels: Optional[Tuple[int]] = None

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)
    

@dataclass
class LorentzParTConfig(BaseModelConfig):
    hidden_mv_channels: int = 8
    in_s_channels: int = None
    out_s_channels: int = None
    hidden_s_channels: int = 16
    attention: Optional[Dict] = None
    mlp: Optional[Dict] = None
    reinsert_mv_channels: Optional[Tuple[int]] = None
    reinsert_s_channels: Optional[Tuple[int]] = None
    pair_embed_dims: List[int] = field(default_factory=lambda: [64, 64, 64])

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)