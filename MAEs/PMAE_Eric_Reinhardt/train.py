import logging
import os
from typing import List, Optional, Sequence, Tuple

import torch

from models.masks import KinematicMask, ParticleMask, SpecificParticleMask
from validate import validate


logger = logging.getLogger(__name__)


SUPPORTED_MODEL_TYPES = {
    "autoencoder",
    "classifier partial",
    "classifier full",
}


def get_mask_layer(
    mask: Optional[int],
    output_vars: int,
    particle_idx: Optional[int] = None,
):
    """Create and return the appropriate input masking layer.

    Args:
        mask: Mask configuration. None disables masking, 0 enables
            particle masking, and any other value enables kinematic masking.
        output_vars: Number of output variables used by the model.
        particle_idx: Specific particle index for full-event classifier
            masking.

    Returns:
        The configured mask layer, or None when masking is disabled.
    """
    if mask is None:
        return None

    mask_dim = output_vars + (output_vars % 3)

    if mask == 0:
        if particle_idx is not None:
            return SpecificParticleMask(mask_dim, particle_idx)
        return ParticleMask(mask_dim)

    return KinematicMask(mask)


def apply_trivial_resets(
    outputs: torch.Tensor,
    masked_inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply the original PMAE trivial-value reset logic.

    The classifier training code uses the value 999 to identify masked
    particle information. The corresponding output features are converted
    to probabilities and reset according to the original training behavior.

    Args:
        outputs: Autoencoder output tensor.
        masked_inputs: Masked input tensor used to determine masked values.

    Returns:
        Processed autoencoder output tensor.
    """
    mask_999 = (masked_inputs[:, :, 3] == 999).float()

    outputs[:, :, 3:5] = torch.nn.functional.softmax(
        outputs[:, :, 3:5],
        dim=2,
    )

    outputs[:, :, 3] = (
        (1 - mask_999) * outputs[:, :, 3]
        + mask_999 * 1
    )

    outputs[:, :, 4] = (
        (1 - mask_999) * outputs[:, :, 4]
    )

    return outputs


def _get_masked_inputs(
    inputs: torch.Tensor,
    mask: Optional[int],
    output_vars: int,
    particle_idx: Optional[int] = None,
) -> torch.Tensor:
    """Apply the configured mask to input data."""
    mask_layer = get_mask_layer(
        mask=mask,
        output_vars=output_vars,
        particle_idx=particle_idx,
    )

    if mask_layer is None:
        return inputs

    return mask_layer(inputs)


def _train_autoencoder_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    tae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    output_vars: int,
    zero_padded: Sequence[int],
    mask: Optional[int],
) -> float:
    """Train the autoencoder for one batch."""
    inputs, _ = batch
    inputs = inputs.to(device)

    masked_inputs = _get_masked_inputs(
        inputs=inputs,
        mask=mask,
        output_vars=output_vars,
    )

    optimizer.zero_grad()

    outputs = tae(masked_inputs)
    outputs = outputs.reshape(
        outputs.size(0),
        outputs.size(1) * outputs.size(2),
    )

    if output_vars == 3:
        targets = inputs[:, :, :-1]
        targets = targets.reshape(
            targets.size(0),
            targets.size(1) * targets.size(2),
        )
        loss = criterion.compute_loss(
            outputs,
            targets,
            zero_padded=[4],
        )

    elif output_vars == 4:
        targets = inputs.reshape(
            inputs.size(0),
            inputs.size(1) * inputs.size(2),
        )
        loss = criterion.compute_loss(
            outputs,
            targets,
            zero_padded=list(zero_padded),
        )

    else:
        raise ValueError(
            f"Unsupported output_vars={output_vars} for autoencoder. "
            "Expected 3 or 4."
        )

    loss.backward()
    optimizer.step()

    return loss.item()


def _train_partial_classifier_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    tae: torch.nn.Module,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    output_vars: int,
    mask: Optional[int],
) -> float:
    """Train the partial-event classifier for one batch."""
    inputs, labels = batch

    inputs = inputs.to(device)
    labels = labels.to(device)

    masked_inputs = _get_masked_inputs(
        inputs=inputs,
        mask=mask,
        output_vars=output_vars,
    )

    optimizer.zero_grad()

    outputs = tae(masked_inputs)

    outputs = apply_trivial_resets(
        outputs,
        masked_inputs,
    )

    outputs = outputs.reshape(
        outputs.size(0),
        outputs.size(1) * outputs.size(2),
    )

    masked_inputs = masked_inputs.reshape(
        masked_inputs.size(0),
        masked_inputs.size(1) * masked_inputs.size(2),
    )

    classifier_inputs = torch.cat(
        (outputs, masked_inputs),
        dim=1,
    )

    predictions = classifier(classifier_inputs).squeeze(1)

    loss = criterion(
        predictions,
        labels.float(),
    )

    loss.backward()
    optimizer.step()

    return loss.item()


def _train_full_classifier_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    tae: torch.nn.Module,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    output_vars: int,
    mask: Optional[int],
) -> float:
    """Train the full-event classifier for one batch."""
    inputs, labels = batch

    inputs = inputs.to(device)
    labels = labels.to(device)

    output_dim = output_vars + (output_vars % 3)

    outputs = torch.zeros(
        inputs.size(0),
        6,
        output_dim,
        device=device,
    )

    masked_inputs = inputs

    optimizer.zero_grad()

    for particle_idx in range(6):
        masked_inputs = _get_masked_inputs(
            inputs=inputs,
            mask=mask,
            output_vars=output_vars,
            particle_idx=particle_idx,
        )

        temp_outputs = tae(masked_inputs)

        outputs[:, particle_idx, :] = (
            temp_outputs[:, particle_idx, :]
        )

    outputs = apply_trivial_resets(
        outputs,
        masked_inputs,
    )

    outputs = outputs.reshape(
        outputs.size(0),
        outputs.size(1) * outputs.size(2),
    )

    inputs = inputs.reshape(
        inputs.size(0),
        inputs.size(1) * inputs.size(2),
    )

    classifier_inputs = torch.cat(
        (outputs, inputs),
        dim=1,
    )

    predictions = classifier(classifier_inputs).squeeze(1)

    loss = criterion(
        predictions,
        labels.float(),
    )

    loss.backward()
    optimizer.step()

    return loss.item()


def _train_epoch(
    train_loader,
    models: List[torch.nn.Module],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion,
    model_type: str,
    output_vars: int,
    zero_padded: Sequence[int],
    mask: Optional[int],
) -> float:
    """Train all batches for one epoch."""
    if model_type == "autoencoder":
        tae = models[0]
        tae.train()

        batch_trainer = _train_autoencoder_batch

    elif model_type == "classifier partial":
        tae, classifier = models[0], models[1]

        tae.eval()
        classifier.train()

        batch_trainer = _train_partial_classifier_batch

    elif model_type == "classifier full":
        tae, classifier = models[0], models[1]

        tae.eval()
        classifier.train()

        batch_trainer = _train_full_classifier_batch

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported values are: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        if model_type == "autoencoder":
            loss = batch_trainer(
                batch=batch,
                tae=tae,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                output_vars=output_vars,
                zero_padded=zero_padded,
                mask=mask,
            )

        else:
            loss = batch_trainer(
                batch=batch,
                tae=tae,
                classifier=classifier,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                output_vars=output_vars,
                mask=mask,
            )

        running_loss += loss

        if (batch_idx + 1) % 500 == 0:
            logger.info(
                "Batch [%d/%d], Loss: %.4f",
                batch_idx + 1,
                len(train_loader),
                running_loss / 500,
            )
            running_loss = 0.0

    return running_loss


def train(
    train_loader,
    val_loader,
    models: List[torch.nn.Module],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion,
    model_type: str,
    output_vars: int,
    zero_padded: Optional[Sequence[int]] = None,
    mask: Optional[int] = None,
    epochs: Optional[Sequence[int]] = None,
    loss_min: float = 999.0,
    save_path: str = "./saved_models",
    model_name: str = "",
) -> float:
    """Train an autoencoder or classifier model.

    Supported model types:
        - ``autoencoder``
        - ``classifier partial``
        - ``classifier full``

    Args:
        train_loader: Training data loader.
        val_loader: Validation data loader.
        models: Models involved in training.
        device: Torch device used for training.
        optimizer: Optimizer used for parameter updates.
        criterion: Loss function.
        model_type: Type of model being trained.
        output_vars: Number of output variables.
        zero_padded: Indices used for autoencoder loss handling.
        mask: Mask configuration.
        epochs: Epoch indices to train.
        loss_min: Initial minimum validation loss.
        save_path: Directory used for saved models.
        model_name: Model name used for output organization.

    Returns:
        The minimum validation loss returned by ``validate``.

    Raises:
        ValueError: If the model type or model configuration is invalid.
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported values are: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    if epochs is None:
        epochs = range(1)

    if zero_padded is None:
        zero_padded = []

    epochs = list(epochs)

    if len(epochs) == 0:
        logger.warning("Number of epochs is 0. Nothing to train.")
        return loss_min

    required_models = 1 if model_type == "autoencoder" else 2

    if len(models) < required_models:
        raise ValueError(
            f"Model type '{model_type}' requires at least "
            f"{required_models} model(s), but received {len(models)}."
        )

    os.makedirs(
        os.path.join("./outputs", model_name),
        exist_ok=True,
    )

    logger.info(
        "Starting training: model_type=%s, epochs=%d, "
        "start_epoch=%d, end_epoch=%d",
        model_type,
        len(epochs),
        epochs[0] + 1,
        epochs[-1] + 1,
    )

    for epoch in epochs:
        logger.info(
            "Starting epoch [%d/%d]",
            epoch + 1,
            epochs[-1] + 1,
        )

        _train_epoch(
            train_loader=train_loader,
            models=models,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            model_type=model_type,
            output_vars=output_vars,
            zero_padded=zero_padded,
            mask=mask,
        )

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
            model_name,
        )

        logger.info(
            "Completed epoch [%d/%d], validation loss minimum: %.6f",
            epoch + 1,
            epochs[-1] + 1,
            loss_min,
        )

    logger.info(
        "Training completed for model_type=%s",
        model_type,
    )

    return loss_min
