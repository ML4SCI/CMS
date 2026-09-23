import torch
from torch import nn


def load_encoder_weights(encoder: nn.Module, weights: str) -> None:
    """
    Load the `encoder.*` tensors of a saved model into `encoder` (non-strict).

    Accepts a best-model state_dict or a trainer checkpoint. Tensors that do not
    match are reported, so e.g. gate weights missing from an ungated pretraining
    run are visible instead of being silently left at their initial values.
    """
    state_dict = torch.load(weights, map_location='cpu')
    if 'model_state_dict' in state_dict:  # trainer checkpoint
        state_dict = state_dict['model_state_dict']

    filtered_state = {
        k[len("encoder.") :]: v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    result = encoder.load_state_dict(filtered_state, strict=False)

    loaded = len(filtered_state) - len(result.unexpected_keys)
    print(f"[weights] {weights}: loaded {loaded} encoder tensors, "
          f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
    if result.missing_keys:
        print(f"[weights]   missing (left at init), e.g. {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"[weights]   unexpected (ignored), e.g. {result.unexpected_keys[:5]}")
