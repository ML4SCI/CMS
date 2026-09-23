"""Distributed training harness for the ParT architecture ablation.

Six arms, one delta each, all trained under an identical optimizer, schedule and
data pipeline so that differences are attributable to the architecture:

===========  ===================================================================
``baseline``   stock weaver ``ParticleTransformer``
``lloca``      Lorentz Local Canonicalization (learned local frames + tensorial
               message passing), arXiv:2505.20280
``n8``         factorized pair bias (Minkowski ``d_ij`` + tensor-product rotary)
``moe``        Mixture-of-Experts FFN, FLOP-matched to the dense baseline
``sparsemax``  sparsemax in place of softmax attention
``diff_v1``    Differential Attention, arXiv:2410.05258
``diff_v2``    Differential Attention V2
===========  ===================================================================

Modules
-------
``config``        the run configuration dataclass, YAML + CLI overrides
``data``         distributed loaders over ragged CSR ``.pt`` shards
``distributed``  Slurm/torchrun process-group setup
``metrics``      accuracy, AUC and background rejection at fixed efficiency
``schedule``     the official ParT optimizer and LR schedule
``train``        the training entrypoint (``python -m ablation.train``)
"""

from .config import ARMS, AblationConfig, load_config

__all__ = ["ARMS", "AblationConfig", "load_config"]
