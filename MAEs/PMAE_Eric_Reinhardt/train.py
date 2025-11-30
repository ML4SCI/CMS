import torch
from validate import validate
from models.masks import ParticleMask, SpecificParticleMask, KinematicMask
from argparse import ArgumentParser
import os

def train(train_loader, val_loader, models, device, optimizer, criterion, model_type, output_vars, zero_padded=[], mask=None, epochs:range=None, loss_min:int=999, save_path:str='./saved_models', model_name:str=''):
    # Create an outputs folder to store config files
    os.makedirs('./outputs/' + model_name, exist_ok=True)
    if len(epochs) <= 0:
        print("Num epochs <= 0")
        return 0
    if model_type == 'autoencoder':
        tae = models[0]
        for epoch in epochs:
            tae.train()
            running_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                # Move the data to the device
                inputs, _ = batch
                inputs = inputs.to(device)
                if mask is not None:
                    if mask == 0:
                        mask_layer = ParticleMask(output_vars+(output_vars%3))
                    else:
                        mask_layer = KinematicMask(mask)
                    # Mask input data
                    masked_inputs = mask_layer(inputs)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = tae(masked_inputs)

                outputs = torch.reshape(outputs, (outputs.size(0),
                                                  outputs.size(1) * outputs.size(2)))

                # Flatten last axes and compute loss
                if output_vars == 3:
                    inputs = inputs[:,:,:-1]
                    inputs = torch.reshape(inputs, (inputs.size(0),
                                                    inputs.size(1) * inputs.size(2)))
                    loss = criterion.compute_loss(outputs, inputs, zero_padded=[4])
                elif output_vars == 4:
                    inputs = torch.reshape(inputs, (inputs.size(0),
                                                    inputs.size(1) * inputs.size(2)))
                    loss = criterion.compute_loss(outputs, inputs, zero_padded=zero_padded)

                # Backward pass
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Print running loss every 500 batches
                if (batch_idx + 1) % 500 == 0:
                    print(f"Epoch [{epoch+1}/{epochs[-1] + 1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / 500:.4f}")
                    running_loss = 0.0

            loss_min = validate(val_loader, models, device, criterion, model_type, output_vars, mask, epoch, epochs[-1] + 1, loss_min, save_path, model_name)
        return loss_min

    elif model_type == 'classifier partial':
        tae, classifier = models[0], models[1]
        for epoch in epochs:
            tae.eval()
            classifier.train()
            running_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                # Move the data to the device
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                if mask is not None:
                    if mask == 0:
                        mask_layer = ParticleMask(output_vars+(output_vars%3))
                    else:
                        mask_layer = KinematicMask(mask)
                    # Mask input data
                    masked_inputs = mask_layer(inputs)

                # Forward pass for autoencoder
                outputs = tae(masked_inputs)

                # Reset trivial values
                mask_999 = (masked_inputs[:, :, 3] == 999).float()
                outputs[:,:,3:5] = torch.nn.functional.softmax(outputs[:,:,3:5], dim=2)
                outputs[:, :, 3] = (1 - mask_999) * outputs[:, :, 3] + mask_999 * 1
                outputs[:, :, 4] = (1 - mask_999) * outputs[:, :, 4]
                masked_inputs[:,:,3:5] = torch.nn.functional.softmax(masked_inputs[:,:,3:5], dim=2)
                masked_inputs[:, :, 3] = (1 - mask_999) * masked_inputs[:, :, 3] + mask_999 * 1
                masked_inputs[:, :, 4] = (1 - mask_999) * masked_inputs[:, :, 4]

                # Flatten last axis
                outputs = torch.reshape(outputs, (outputs.size(0),
                                                    outputs.size(1) * outputs.size(2)))

                masked_inputs = torch.reshape(masked_inputs, (masked_inputs.size(0),
                                                                masked_inputs.size(1) * masked_inputs.size(2)))

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass for classifier
                outputs_2 = classifier(torch.cat((outputs, masked_inputs), axis=1)).squeeze(1)

                # Caclulate the loss
                loss = criterion(outputs_2, labels.float())

                # Backward pass
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Print running loss every 500 batches
                if (batch_idx + 1) % 500 == 0:
                    print(f"Epoch [{epoch+1}/{epochs[-1] + 1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / 500:.4f}")
                    running_loss = 0.0

            loss_min = validate(val_loader, models, device, criterion, model_type, output_vars, mask, epoch, epochs[-1] + 1, loss_min, save_path, model_name)
        return loss_min
        
    elif model_type == 'classifier full':
        tae, classifier = models[0], models[1]
        for epoch in epochs:
            tae.eval()
            classifier.train()
            running_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                # Move the data to the device
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = torch.zeros(inputs.size(0), 6, output_vars+(output_vars%3)).to(device)
                for i in range(6):
                    if mask is not None:
                        if mask == 0:
                            mask_layer = SpecificParticleMask(output_vars+(output_vars%3), i)
                        else:
                            mask_layer = KinematicMask(mask)
                        # Mask input data
                        masked_inputs = mask_layer(inputs)
                    # Forward pass for autoencoder
                    temp_outputs = tae(masked_inputs)
                    outputs[:,i,:] = temp_outputs[:,i,:]

                # Reset trivial values
                mask_999 = (masked_inputs[:, :, 3] == 999).float()
                outputs[:,:,3:5] = torch.nn.functional.softmax(outputs[:,:,3:5], dim=2)
                outputs[:, :, 3] = (1 - mask_999) * outputs[:, :, 3] + mask_999 * 1
                outputs[:, :, 4] = (1 - mask_999) * outputs[:, :, 4]

                # Flatten last axes of tensors
                outputs = torch.reshape(outputs, (outputs.size(0),
                                                    outputs.size(1) * outputs.size(2)))

                inputs = torch.reshape(inputs, (inputs.size(0),
                                                inputs.size(1) * inputs.size(2)))

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass for classifier
                outputs_2 = classifier(torch.cat((outputs, inputs), axis=1)).squeeze(1)

                # Caclulate the loss
                loss = criterion(outputs_2, labels.float())

                # Backward pass
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Print running loss every 500 batches
                if (batch_idx + 1) % 500 == 0:
                    print(f"Epoch [{epoch+1}/{epochs[-1] + 1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / 500:.4f}")
                    running_loss = 0.0

            loss_min = validate(val_loader, models, device, criterion, model_type, output_vars, mask, epoch, epochs[-1] + 1, loss_min, save_path, model_name)
        return loss_min