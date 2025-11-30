import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import json
from torch import optim, nn
import torch
from sklearn.metrics import f1_score
import scipy

def optimize_thresholds(y_true, y_pred, mask=None, epsilon=.2):
    y_t = y_true.copy() + 1
    y_p = y_pred.copy() + 1

    if len(y_t[y_t != 0]) == 0:
        return np.zeros_like(y_p) - 1

    sorted_y_t = y_t[~mask]
    sorted_y_p = y_p[~mask]
    sorted_indices = np.argsort(sorted_y_p)
    sorted_y_t = sorted_y_t[sorted_indices]
    sorted_y_p = sorted_y_p[sorted_indices]

    def objective(threshold, sorted_y_t, sorted_y_p):
        classified_preds = np.zeros_like(sorted_y_t)
        classified_preds[sorted_y_p > threshold + epsilon] = 2
        classified_preds[sorted_y_p < threshold - epsilon] = 0
        classified_preds = classified_preds[sorted_y_t != 1]
        sorted_y_t = sorted_y_t[sorted_y_t != 1]
        f1 = f1_score(sorted_y_t, classified_preds, average='weighted')
        return -f1

    initial_threshold = [1]
    bounds = [(.25, 1.75)]
    
    result = scipy.optimize.minimize(objective, initial_threshold, bounds=bounds, method='L-BFGS-B', 
                                     args=(sorted_y_t, sorted_y_p))

    optimized_threshold = result.x[0]

    y_p[y_p > optimized_threshold + epsilon] = 2
    y_p[(y_p <= optimized_threshold + epsilon) & (y_p >= optimized_threshold - epsilon)] = 1
    y_p[y_p < optimized_threshold - epsilon] = 0

    return y_p - 1

def make_hist2d(group_num, steps, ins, outs, scaler, event_type, file_path, mask=None, lower=None, upper=None):
    names = ["lepton pT", "lepton eta", "lepton phi", "Padding",
             "missing energy magnitude", "Padding", "missing energy phi", "Padding",
             "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag",
             "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag",
             "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag",
             "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag"]


    inputs = scaler.inverse_transform(ins)
    outputs = scaler.inverse_transform(outs)

    if steps == 4:
        inputs[:,3::4] = ins[:,3::4]
        outputs[:,3::4] = outs[:,3::4]

    for step in range(steps):
        if step == 3:
            bins = 30
            varname = names[group_num*steps+step]
            heatmap, xedges, yedges = np.histogram2d(inputs[:,group_num*steps+step],
                                                    outputs[:,group_num*steps+step],
                                                    bins=bins,
                                                    range=[[lower[step], upper[step]], [lower[step], upper[step]]])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            #Plot heatmap
            plt.imshow(heatmap.T,
                    extent=extent,
                    origin='lower')
            plt.plot([lower[step], upper[step]],
                    [lower[step], upper[step]],
                    color='blue')
            fig = plt.gcf()
            plt.set_cmap('gist_heat_r')
            plt.xlabel('%s scaled True' % varname)
            plt.ylabel('%s scaled Pred' % varname)
            plt.title('Frequency Heatmap (' + event_type + ')')
            plt.xlim(lower[step], upper[step])
            plt.ylim(lower[step], upper[step])
            plt.colorbar()
            plt.savefig(file_path + '/hist2d_' + event_type + '_' + names[group_num*steps+step] + '_high_res.png')
            plt.show()
            plt.close()
        if step == 3:
            bins = 3
            outputs[:, group_num*steps+step] = optimize_thresholds(inputs[:,group_num*steps+step], outputs[:,group_num*steps+step], mask=mask)
        else:
            bins = 30
        varname = names[group_num*steps+step]
        heatmap, xedges, yedges = np.histogram2d(inputs[:,group_num*steps+step],
                                                 outputs[:,group_num*steps+step],
                                                 bins=bins,
                                                 range=[[lower[step], upper[step]], [lower[step], upper[step]]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        #Plot heatmap
        plt.imshow(heatmap.T,
                   extent=extent,
                   origin='lower')
        plt.plot([lower[step], upper[step]],
                 [lower[step], upper[step]],
                 color='blue')
        fig = plt.gcf()
        plt.set_cmap('gist_heat_r')
        plt.xlabel('%s scaled True' % varname)
        plt.ylabel('%s scaled Pred' % varname)
        plt.title('Frequency Heatmap (' + event_type + ')')
        plt.xlim(lower[step], upper[step])
        plt.ylim(lower[step], upper[step])
        plt.colorbar()
        plt.savefig(file_path + '/hist2d_' + event_type + '_' + names[group_num*steps+step] + '.png')
        plt.show()
        plt.close()

class SoftLabelFocalLoss(nn.Module):
    def __init__(self, gamma=2., reduction='mean', entropy_weight=.2):
        super(SoftLabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.entropy_weight = entropy_weight

    def forward(self, inputs, targets, alpha):
        batch_size = len(inputs)

        mask = targets[:, 0] != 999

        inputs = inputs[mask]
        targets = targets[mask]

        if targets.shape[0] == 0:
            return torch.tensor(0.0).to(targets.device)

        alpha = torch.from_numpy(alpha).to(inputs.device).float()
        probs = nn.functional.softmax(inputs, dim=1)
        fl = -alpha * (targets * (1. - probs).pow(self.gamma) * torch.log(probs + 1e-6) +\
         (1. - targets) * probs.pow(self.gamma) * torch.log(1. - probs + 1e-6))
        fl = fl * targets
        fl = fl.sum(dim=1)

        entropy = -(probs.exp() * probs).sum(dim=1)
        fl += self.entropy_weight * entropy

        if self.reduction == 'mean':
            return fl.mean() * len(inputs) / batch_size
        elif self.reduction == 'sum':
            return fl.sum() * len(inputs) / batch_size
        else:
            return fl

# Custom loss functions
class custom_loss:
    def __init__(self, phi_limit, alpha=0.4, beta=.5, gamma=1., delta=.5, lower_pt_limit=[], f_alphas=[], output_vars=3):
        self.phi_limit = phi_limit
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.f_alphas = f_alphas
        self.vars = output_vars
        self.lower_pt_limit = lower_pt_limit

    def compute_loss(self, output, target, zero_padded=[]):
        loss = 0
        if self.vars == 3:
            for i in range(output.size()[1] - 2):
                if i in zero_padded:
                    continue
                elif i % self.vars == 0:
                    loss += torch.mean((target[:,i] - output[:,i])**2 + torch.gt(output[:,i], self.lower_pt_limit[i // 4]).long() * \
                        (self.gamma / (1 + torch.exp(-(output[:,i] - self.lower_pt_limit[i // 4]) * 3)) - self.gamma) + \
                            torch.le(output[:,i], self.lower_pt_limit[i // 4]).long()*(self.gamma/2 - self.gamma))
                elif i % self.vars == 1:
                    loss += torch.mean((target[:,i] - output[:,i])**2 - output[:,i]**2 * self.beta)
                elif i % self.vars == 2:
                    loss += torch.mean(torch.le(torch.abs(output[:,i]), self.phi_limit).long() *\
                        ((torch.sin(((output[:,i] - target[:,i]) / self.phi_limit - .5) * np.pi) + 1)**2 +\
                            (torch.sin(((output[:,i] - target[:,i]) / self.phi_limit - .5) * np.pi) + 1) * 2) * self.alpha +\
                        torch.gt(torch.abs(output[:,i]), self.phi_limit).long() *\
                        (((torch.sin(((self.phi_limit * torch.sign(output[:,i]) - target[:,i]) / self.phi_limit  - .5) * \
                                        np.pi) + 1)**2 +\
                            (torch.sin(((self.phi_limit * torch.sign(output[:,i]) - target[:,i]) / self.phi_limit  - .5) * \
                                    np.pi) + 1) * 2) * self.alpha +\
                        (self.phi_limit*torch.sign(output[:,i]) - output[:,i])**2))
            return loss / (output.size()[1] - len(zero_padded))
        else:
            self.vars = 5
            for i in range(output.size()[1]-2):
                if i in zero_padded:
                    continue
                elif i % self.vars == 0:
                    loss += torch.mean((target[:,i] - output[:,i])**2 + torch.gt(output[:,i], self.lower_pt_limit[i // self.vars]).long() * \
                        (self.gamma / (1 + torch.exp(-(output[:,i] - self.lower_pt_limit[i // self.vars]) * 3)) - self.gamma) + \
                            torch.le(output[:,i], self.lower_pt_limit[i // self.vars]).long()*(self.gamma/2 - self.gamma))
                elif i % self.vars == 1:
                    loss += torch.mean((target[:,i] - output[:,i])**2 - output[:,i]**2 * self.beta)
                elif i % self.vars == 2:
                    loss += torch.mean(torch.le(torch.abs(output[:,i]), self.phi_limit).long() *\
                        ((torch.sin(((output[:,i] - target[:,i]) / self.phi_limit - .5) * np.pi) + 1)**2 +\
                            (torch.sin(((output[:,i] - target[:,i]) / self.phi_limit - .5) * np.pi) + 1) * 2) * self.alpha +\
                        torch.gt(torch.abs(output[:,i]), self.phi_limit).long() *\
                        (((torch.sin(((self.phi_limit * torch.sign(output[:,i]) - target[:,i]) / self.phi_limit  - .5) * \
                                        np.pi) + 1)**2 +\
                            (torch.sin(((self.phi_limit * torch.sign(output[:,i]) - target[:,i]) / self.phi_limit  - .5) * \
                                    np.pi) + 1) * 2) * self.alpha +\
                        (self.phi_limit*torch.sign(output[:,i]) - output[:,i])**2))
                elif i % self.vars == 3:
                    loss += (SoftLabelFocalLoss()(output[:,i:i+2], target[:,i:i+2], self.f_alphas[(i - 3) // self.vars])) * self.delta
            return loss / (output.size()[1] - len(zero_padded) - 6)

# Dataset class
class DataLabelDataset(Dataset):
    def __init__(self, data, labels, dtype: str = 'numpy'):
        super(DataLabelDataset, self).__init__()
        if dtype == 'numpy':
            self.data = torch.from_numpy(data).type(torch.FloatTensor)
            self.labels = torch.from_numpy(labels).type(torch.FloatTensor)
        elif dtype == 'torch':
            self.data = data
            self.labels = labels
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

# Custom SGD optimizer
class SGDWithSaturatingMomentumAndDecay(optim.Optimizer):
    def __init__(self, params, lr=None, momentum=0, max_momentum=0.99, epochs_to_saturate=100, batches_per_epoch=1, weight_decay=0, lr_decay=0.1, min_lr=1e-6, resume_epoch=0):
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, max_momentum=max_momentum, epochs_to_saturate=epochs_to_saturate, batches_per_epoch=batches_per_epoch, weight_decay=weight_decay, lr_decay=lr_decay, min_lr=min_lr, resume_epoch=resume_epoch)
        super(SGDWithSaturatingMomentumAndDecay, self).__init__(params, defaults)

        for group in self.param_groups:
            # Adjust initial learning rate and momentum based on resume epoch
            steps_to_saturate = group['epochs_to_saturate'] * group['batches_per_epoch']
            resumed_steps = group['resume_epoch'] * group['batches_per_epoch']
            max_momentum = group['max_momentum']
            momentum_step = (max_momentum - group['momentum']) / steps_to_saturate
            group['momentum'] = min(group['momentum'] + momentum_step * resumed_steps, max_momentum)
            group['lr'] = max(group['lr'] * (group['lr_decay'] ** resumed_steps), group['min_lr'])

    def step(self, closure=None):
        for group in self.param_groups:
            steps_to_saturate = group['epochs_to_saturate'] * group['batches_per_epoch']
            max_momentum = group['max_momentum']
            momentum_step = (max_momentum - group['momentum']) / steps_to_saturate

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if group['weight_decay'] != 0:
                    d_p.add_(p.data, alpha=group['weight_decay'])

                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

            # Increment momentum and decay learning rate after the step
            group['momentum'] = min(group['momentum'] + momentum_step, max_momentum)
            group['lr'] = max(group['lr'] * group['lr_decay'], group['min_lr'])

def parse_model_name(model_name):
    data = {}

    # A dictionary to map from the keys in the model name to the keys in the JSON
    key_map = {
        "DM": "d_model",
        "H": "num_heads",
        "L": "num_layers",
        "F": "d_ff",
        "DR": "dropout",
        "BS": "batch_size",
        "T": "test_batch_size",
        "AE": "ae_resume_epoch",
        "PC": "pc_resume_epoch",
        "FC": "fc_resume_epoch",
        "ANE": "ae_num_epochs",
        "PNE": "pc_num_epochs",
        "FNE": "fc_num_epochs",
        "AES": "ae_epochs_to_saturate",
        "PES": "pc_epochs_to_saturate",
        "FES": "fc_epochs_to_saturate",
        "IM": "init_momentum",
        "MM": "max_momentum",
        "TILR": "tae_init_lr",
        "PCLR": "pc_init_lr",
        "FCLR": "fc_init_lr",
        "MSL": "max_seq_len",
        "Mk": "mask",
        "A": "alpha",
        "B": "beta",
        "G": "gamma",
        "D": "delta",
        "OV": "output_vars",
        "WD": "weight_decay",
        "MLR": "min_lr",
        "ALD": "ae_lr_decay",
        "PLD": "pc_lr_decay",
        "FLD": "fc_lr_decay",
        "CIF": "class_input_features",
        "CFD": "class_ff_dim"
    }

    # Remove 'Model' from the start of the model name
    model_name = model_name.lstrip('Model_')

    # Iterate through each key in the key map
    for key in key_map.keys():
        # If the model name contains the key
        if key in model_name:
            if key in ["AE", "PC", "FC"]:
                data[key_map[key]] = True
                continue
            # Find the start and end index of the value
            start = model_name.index(key) + len(key)
            end = model_name.index('_', start) if '_' in model_name[start:] else len(model_name)
            
            # Extract and convert the value
            value = model_name[start:end]
            if 'e' in value or '.' in value:  # The value is a float
                value = float(value)
            else:
                value = int(value)
            
            # Add the key-value pair to the dictionary
            data[key_map[key]] = value
            
            # Remove the processed part from the model name
            model_name = model_name[end+1:]

    return data