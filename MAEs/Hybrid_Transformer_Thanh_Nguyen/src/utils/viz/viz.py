from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# Function to visualize the distribution of particle features (pT, eta, phi, E)
def plot_feature_distribution(X_jets: np.ndarray) -> None:
    feature_names = ['pT', 'eta', 'phi', 'energy']
    plt.figure(figsize=(12, 8))

    for i, feature in enumerate(feature_names):
        plt.subplot(2, 2, i + 1)
        sns.histplot(X_jets[:, i], bins=50, kde=True)
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# Function to visualize the particle features reconstruction
def plot_particle_reconstruction(y_true: np.ndarray, y_pred: np.ndarray, save_fig: Optional[str] = None) -> None:
    pT_true, eta_true, phi_true, E_true = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    pT_pred, eta_pred, phi_pred, E_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

    plt.figure(figsize=(12, 10))

    # pT histogram
    pT_min = min(pT_true.min(), pT_pred.min())
    pT_max = max(pT_true.max(), pT_pred.max())
    plt.subplot(2, 2, 1)
    plt.hist2d(pT_true, pT_pred, bins=50, cmap='gist_heat_r')
    plt.xlabel("true scaled pT")
    plt.ylabel("predicted scaled pT")
    plt.title("scaled pT distribution")
    plt.colorbar()
    plt.plot([pT_min, pT_max], [pT_min, pT_max], color='blue', linestyle='-')
    plt.xlim(pT_min, pT_max)
    plt.ylim(pT_min, pT_max)
    
    # Eta histogram
    eta_min = min(eta_true.min(), eta_pred.min())
    eta_max = max(eta_true.max(), eta_pred.max())
    plt.subplot(2, 2, 2)
    plt.hist2d(eta_true, eta_pred, bins=50, cmap='gist_heat_r')
    plt.xlabel("true eta")
    plt.ylabel("predicted eta")
    plt.title("eta distribution")
    plt.colorbar()
    plt.plot([eta_min, eta_max], [eta_min, eta_max], color='blue', linestyle='-')
    plt.xlim(eta_min, eta_max)
    plt.ylim(eta_min, eta_max)
    
    # Phi histogram
    phi_min = min(phi_true.min(), phi_pred.min())
    phi_max = max(phi_true.max(), phi_pred.max())
    plt.subplot(2, 2, 3)
    plt.hist2d(phi_true, phi_pred, bins=50, cmap='gist_heat_r')
    plt.xlabel("true phi")
    plt.ylabel("predicted phi")
    plt.title("phi distribution")
    plt.colorbar()
    plt.plot([phi_min, phi_max], [phi_min, phi_max], color='blue', linestyle='-')
    plt.xlim(phi_min, phi_max)
    plt.ylim(phi_min, phi_max)

    # Energy histogram
    E_min = min(E_true.min(), E_pred.min())
    E_max = max(E_true.max(), E_pred.max())
    plt.subplot(2, 2, 4)
    plt.hist2d(E_true, E_pred, bins=50, cmap='gist_heat_r')
    plt.xlabel("true scaled energy")
    plt.ylabel("predicted scaled energy")
    plt.title("scaled energy distribution")
    plt.colorbar()
    plt.plot([E_min, E_max], [E_min, E_max], color='blue', linestyle='-')
    plt.xlim(E_min, E_max)
    plt.ylim(E_min, E_max)
    
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=300)
    else:
        plt.show()


# Function to visualize the training progress
def plot_history(history: Dict[str, List[float]], save_fig: Optional[str] = None) -> None:
    plt.figure(figsize=(12, 5))
    epochs = history['epoch']

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['val_loss'], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot training and validation metric (accuracy, for example)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_metric'], label="Train Accuracy")
    plt.plot(epochs, history['val_metric'], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=300)
    else:
        plt.show()


# Function to visualize the self-supervised masked model training progress
def plot_ssl_history(history: Dict[str, List[float]], save_fig: Optional[str] = None) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(history['pT_loss'], label="pT_loss")
    plt.plot(history['eta_loss'], label="eta_loss")
    plt.plot(history['phi_loss'], label="phi_loss")
    plt.plot(history['energy_loss'], label="energy_loss")
    plt.plot(history['val_loss'], label="val_loss")
    plt.title("Self-supervised Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=300)
    else:
        plt.show()


# Function to visualize the confusion matrix
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None, save_fig: Optional[str] = None) -> None:
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes, labels=np.arange(y_true.shape[1]))
    cm = cm / 1000

    if labels is None:
        labels = [
            "$q/g$",  # 0
            "$H \\to b\\bar{b}$",  # 1
            "$H \\to c\\bar{c}$",  # 2
            "$H \\to gg$",  # 3
            "$H \\to 4q$",  # 4
            "$H \\to \\ell \\nu qq'$",  # 5
            "$Z \\to q\\bar{q}$",  # 6
            "$W \\to qq'$",  # 7
            "$t \\to b\\ell \\nu$",  # 8
            "$t \\to bqq'$"  # 9
        ]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        data=cm,
        annot=True,
        fmt='.1f',
        annot_kws={'size': 8},
        xticklabels=labels,
        yticklabels=labels,
        cmap='coolwarm'
    )
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (in thousands)")
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=300)
    else:
        plt.show()


# Function to visualize the ROC curve
def plot_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray, save_fig: Optional[str] = None) -> None:
    # Convert one-hot encoded y_true to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true
    
    # Get number of classes
    n_classes = y_pred_prob.shape[1]
    
    # Compute macro-average ROC AUC score
    roc_auc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovo')
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    for i in range(n_classes):
        # For each class, get binary indicators
        y_true_binary = (y_true_indices == i).astype(int)
        y_score = y_pred_prob[:, i]
        
        # Calculate ROC curve for this class
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score)
    
    # Compute macro-average ROC curve by interpolating and averaging
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Average and compute AUC
    mean_tpr /= n_classes
    
    # Plot the macro-average ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(all_fpr, mean_tpr, color='orange', label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro-Average ROC Curve")
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=300)
    else:
        plt.show()