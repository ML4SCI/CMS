import torch
from validate import validate
from models.masks import ParticleMask, SpecificParticleMask, KinematicMask
import os
import logging
from tqdm import tqdm
from typing import List, Optional

# ------------------------
# 1. Setup Logging
# ------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ------------------------
# 2. Helper Functions
# ------------------------
def get_mask_layer(mask_type: Optional[int], output_vars: int, particle_idx: Optional[int] = None):
    """Return the appropriate mask layer."""
    if mask_type is None:
        return None
    dim = output_vars + (output_vars % 3)
    if mask_type == 0:
        if particle_idx is not None:
            return SpecificParticleMask(dim, particle_idx)
        return ParticleMask(dim)
    return KinematicMask(mask_type)


def apply_trivial_resets(outputs: torch.Tensor, masked_inputs: torch.Tensor) -> torch.Tensor:
    """Apply softmax and reset trivial values for physics."""
    mask_999 = (masked_inputs[:, :, 3] == 999).float()
    outputs[:, :, 3:5] = torch.nn.functional.softmax(outputs[:, :, 3:5], dim=2)
    outputs[:, :, 3] = (1 - mask_999) * outputs[:, :, 3] + mask_999 * 1
    outputs[:, :, 4] = (1 - mask_999) * outputs[:, :, 4]
    return outputs


# ------------------------
# 3. Main Training Function
# ------------------------
def train(
    train_loader,
    val_loader,
    models: List[torch.nn.Module],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion,
    model_type: str,
    output_vars: int,
    zero_padded: List[int] = [],
    mask: Optional[int] = None,
    epochs: range = range(1),
    loss_min: float = 999.0,
    save_path: str = './saved_models',
    model_name: str = ''
) -> float:

    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    if len(epochs) <= 0:
        logger.error("Number of epochs must be greater than 0")
        return 0

    # Determine model handles
    if 'classifier' in model_type:
        tae, classifier = models[0], models[1]
    else:
        tae = models[0]

    # ------------------------
    # Main Epoch Loop
    # ------------------------
    for epoch in epochs:
        running_loss = 0.0

        # Set modes
        if model_type == 'autoencoder':
            tae.train()
        elif 'classifier' in model_type:
            tae.eval()
            classifier.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs[-1]+1}")

        for batch in pbar:
            # Safe unpacking: handles datasets without labels
            if model_type == 'autoencoder':
                inputs = batch[0].to(device)
                labels = None
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            # ------------------------
            # Model Logic
            # ------------------------
            if model_type == 'autoencoder':
                mask_layer = get_mask_layer(mask, output_vars)
                masked_inputs = mask_layer(inputs) if mask_layer else inputs

                outputs = tae(masked_inputs)
                outputs = outputs.flatten(1)

                # Prepare targets
                targets = inputs[:, :, :-1] if output_vars == 3 else inputs
                targets = targets.flatten(1)

                loss = criterion.compute_loss(outputs, targets, zero_padded=zero_padded)

            elif model_type == 'classifier partial':
                mask_layer = get_mask_layer(mask, output_vars)
                masked_inputs = mask_layer(inputs) if mask_layer else inputs

                outputs = apply_trivial_resets(tae(masked_inputs), masked_inputs).flatten(1)
                flat_masked = masked_inputs.flatten(1)

                preds = classifier(torch.cat((outputs, flat_masked), dim=1)).squeeze(1)
                loss = criterion(preds, labels.float())

            elif model_type == 'classifier full':
                batch_size = inputs.size(0)
                dim = output_vars + (output_vars % 3)
                outputs = torch.zeros(batch_size, 6, dim).to(device)

                for i in range(6):
                    mask_layer = get_mask_layer(mask, output_vars, particle_idx=i)
                    masked_inputs = mask_layer(inputs) if mask_layer else inputs
                    temp_outputs = tae(masked_inputs)
                    outputs[:, i, :] = temp_outputs[:, i, :]

                outputs = apply_trivial_resets(outputs, inputs).flatten(1)
                flat_inputs = inputs.flatten(1)

                preds = classifier(torch.cat((outputs, flat_inputs), dim=1)).squeeze(1)
                loss = criterion(preds, labels.float())

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # ------------------------
            # Backward Pass
            # ------------------------
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ------------------------
        # Validation
        # ------------------------
        loss_min = validate(
            val_loader,
            models,
            device,
            criterion,
            model_type,
            output_vars,
            mask,
            epoch,
            epochs[-1] + 1,
            loss_min,
            save_path,
            model_name
        )

    return loss_min
