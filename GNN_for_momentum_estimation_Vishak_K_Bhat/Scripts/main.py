import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import GraphDataset

from models import MODEL_GNN
from utils import save_results, preprocess_data

def train(train_loader, model, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for data in tqdm(train_loader, position=0):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        labels = data.y
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.num_graphs
    return train_loss / len(train_loader.dataset)

def test(test_loader, model, criterion, scaler, device):
    model.eval()
    y_true_all = []
    y_pred_all = []
    test_loss = 0

    with torch.no_grad():
        for data in tqdm(test_loader, position=0):
            data = data.to(device)
            out = model(data)
            labels = data.y
            loss = criterion(out, labels)
            test_loss += loss.item() * data.num_graphs

            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(out.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_true_all = y_true_all * scaler.scale_[-1] + scaler.mean_[-1]
    y_pred_all = y_pred_all * scaler.scale_[-1] + scaler.mean_[-1]

    return test_loss, y_true_all, y_pred_all

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    filtered_df, _, labels, train_idx, test_idx, scaler = preprocess_data(args.input_csv)
    min_pT = labels.min()
    max_pT = labels.max()
    # Define edge index for the graph (complete graph)
    num_nodes = 4
    edge_index = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

    # Create dataset and dataloader
    train_dataset = GraphDataset(filtered_df, labels, edge_index, train_idx, args.node_feat)
    test_dataset = GraphDataset(filtered_df, labels, edge_index, test_idx, args.node_feat)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model, criterion, and optimizer
    model = MODEL_GNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5, verbose=True)

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, optimizer, criterion, device)
        test_loss, y_true, y_pred = test(test_loader, model, criterion, scaler, device)

        lr_scheduler.step(test_loss)
        print(f'Epoch {epoch + 1}/{args.epochs} | Training Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch + 1}.pth'))

            # Save and plot results
            os.makedirs(args.save_dir, exist_ok=True)
            save_results(test_loss, y_true, y_pred, args.save_dir, min_pT, max_pT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Evaluate GNN Model')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--node_feat', type=str, default="bendAngle", choices=['bendAngle', 'etaValue'], help='Which feature you want to make the node feature')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--save_interval', type=int, default=3, help='Interval for saving model checkpoints')
    parser.add_argument('--save_dir', type=str, default='./model_trained', help='Directory to save model checkpoints and results')

    args = parser.parse_args()
    main(args)
