import torch
from models.masks import ParticleMask, SpecificParticleMask, KinematicMask
import json
from utils import parse_model_name
import os

# Validation loop
def validate(val_loader, models, device, criterion, model_type, output_vars, mask, epoch, num_epochs, loss_min, save_path, model_name):
    # Create a config checkpoint file
    config = parse_model_name(model_name)
    if model_type == 'autoencoder':
        dir_name = './outputs/' + model_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.exists(dir_name + '/ckpt_config.json'):
            with open(dir_name + '/ckpt_config.json', 'w') as f:
                json.dump(config, f, indent=4)
        tae = models[0]
        tae.eval()  # Set the tae to evaluation mode
        losses = []
        with torch.no_grad():  # Disable gradient calculations
            for val_batch in val_loader:
                # Move the data to the device
                inputs, _ = val_batch
                inputs = inputs.to(device)
                if mask is not None:
                    if mask == 0:
                        mask_layer = ParticleMask(output_vars+(output_vars%3))
                    else:
                        mask_layer = KinematicMask(mask)
                    # Mask input data
                    masked_inputs = mask_layer(inputs)

                # Forward pass
                outputs = tae(masked_inputs)
                outputs = torch.reshape(outputs, (outputs.size(0),
                                                  outputs.size(1) * outputs.size(2)))

                # Flatten last axes
                if output_vars == 3:
                    inputs = inputs[:,:,:-1]
                    inputs = torch.reshape(inputs, (inputs.size(0),
                                                    inputs.size(1) * inputs.size(2)))
                    loss = criterion.compute_loss(outputs, inputs, zero_padded=[4])
                elif output_vars == 4:
                    inputs = torch.reshape(inputs, (inputs.size(0),
                                                    inputs.size(1) * inputs.size(2)))
                    loss = criterion.compute_loss(outputs, inputs, zero_padded=[3,6,8])

                losses.append(loss.item())

        loss_mean = sum(losses) / len(losses)

        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {loss_mean:.4f}")
        
        # Save files if better than best performance
        if loss_mean < loss_min:
            loss_min = loss_mean
            torch.save(tae.state_dict(), save_path + '/TAE_best_' + model_name)

        # Update the checkpoint file
        with open('./outputs/' + model_name + '/ckpt_config.json', 'r') as f:
            config = json.load(f)
        config['ae_resume_epoch'] = epoch + 1
        with open('./outputs/' + model_name + '/ckpt_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        return loss_min

    elif model_type == 'classifier partial':
        dir_name = './outputs/' + model_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.exists(dir_name + '/ckpt_config.json'):
            with open(dir_name + '/ckpt_config.json', 'w') as f:
                json.dump(config, f, indent=4)
        tae, classifier = models[0], models[1]
        # Validation loop
        tae.eval()  # Set the tae to evaluation mode
        classifier.eval()
        losses = []
        with torch.no_grad():  # Disable gradient calculations
            for val_batch in val_loader:
                # Move the data to the device
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                if mask is not None:
                    if mask == 0:
                        mask_layer = ParticleMask(output_vars+(output_vars%3))
                    else:
                        mask_layer = KinematicMask(mask)
                    # Mask input data
                    masked_inputs = mask_layer(inputs)

                # Forward pass
                outputs = tae(masked_inputs)

                # Reset trivial values
                mask_999 = (masked_inputs[:, :, 3] == 999).float()
                outputs[:,:,3:5] = torch.nn.functional.softmax(outputs[:,:,3:5], dim=2)
                outputs[:, :, 3] = (1 - mask_999) * outputs[:, :, 3] + mask_999 * 1
                outputs[:, :, 4] = (1 - mask_999) * outputs[:, :, 4]
                masked_inputs[:,:,3:5] = torch.nn.functional.softmax(masked_inputs[:,:,3:5], dim=2)
                masked_inputs[:, :, 3] = (1 - mask_999) * masked_inputs[:, :, 3] + mask_999 * 1
                masked_inputs[:, :, 4] = (1 - mask_999) * masked_inputs[:, :, 4]

                outputs = torch.reshape(outputs, (outputs.size(0),
                                                  outputs.size(1) * outputs.size(2)))

                masked_inputs = torch.reshape(masked_inputs, (masked_inputs.size(0),
                                                              masked_inputs.size(1) * masked_inputs.size(2)))

                # Concat encodings and masked inputs
                outputs_2 = classifier(torch.cat((outputs, masked_inputs), axis=1)).squeeze(1)

                loss = criterion(outputs_2, labels.float())
                losses.append(loss.item())
        
        loss_mean = sum(losses) / len(losses)

        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {loss_mean:.4f}")
        
        # Save files if better than best performance
        if loss_mean < loss_min:
            loss_min = loss_mean
            torch.save(classifier.state_dict(), save_path + '/Classifier_partial_best_' + model_name)

        # Update the checkpoint file
        with open('./outputs/' + model_name + '/ckpt_config.json', 'r') as f:
            config = json.load(f)
        config['pc_resume_epoch'] = epoch + 1
        with open('./outputs/' + model_name + '/ckpt_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        return loss_min

    elif model_type == 'classifier full':
        dir_name = './outputs/' + model_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.exists(dir_name + '/ckpt_config.json'):
            with open(dir_name + '/ckpt_config.json', 'w') as f:
                json.dump(config, f, indent=4)
        tae, classifier = models[0], models[1]
        # Validation loop
        tae.eval()  # Set the tae to evaluation mode
        classifier.eval()
        losses = []
        with torch.no_grad():  # Disable gradient calculations
            for val_batch in val_loader:
                # Move the data to the device
                inputs, labels = val_batch
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

                # Flatten last axis
                outputs = torch.reshape(outputs, (outputs.size(0),
                                                  outputs.size(1) * outputs.size(2)))
                inputs = torch.reshape(inputs, (inputs.size(0),
                                                inputs.size(1) * inputs.size(2)))

                # Concat encodings and inputs
                outputs_2 = classifier(torch.cat((outputs, inputs), axis=1)).squeeze(1)

                loss = criterion(outputs_2, labels.float())
                losses.append(loss.item())

        loss_mean = sum(losses) / len(losses)

        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {loss_mean:.4f}")
        
        # Save files if better than best performance
        if loss_mean < loss_min:
            loss_min = loss_mean
            torch.save(classifier.state_dict(), save_path + '/Classifier_full_best_' + model_name)

        # Update the checkpoint file
        with open('./outputs/' + model_name + '/ckpt_config.json', 'r') as f:
            config = json.load(f)
        config['fc_resume_epoch'] = epoch + 1
        with open('./outputs/' + model_name + '/ckpt_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        return loss_min