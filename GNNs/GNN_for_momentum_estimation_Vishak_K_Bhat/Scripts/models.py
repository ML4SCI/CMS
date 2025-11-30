# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GlobalAttention
from torch_geometric.utils import softmax

class MPL(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_attr_dim=3, edge_hidden_dim=16):
        super(MPL, self).__init__(aggr='add')
        self.mlp1 = nn.Linear(in_channels*2 + edge_hidden_dim, out_channels)
        self.mlp2 = nn.Linear(in_channels, out_channels)
        self.mlp3 = nn.Linear(2*out_channels, 1)
        self.mlp4 = nn.Linear(2*out_channels, 1)
        self.mlp5 = nn.Linear(in_channels, 16)
        self.mlp6 = nn.Linear(out_channels, 16)
        self.mlp7 = nn.Linear(16, 1)
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        edge_attr_transformed = self.edge_mlp(edge_attr)
        msg = self.propagate(edge_index, x=x, edge_attr=edge_attr_transformed)
        
        x = F.relu(self.mlp2(x))
        w1 = F.sigmoid(self.mlp3(torch.cat([x, msg], dim=1)))
        w2 = F.sigmoid(self.mlp4(torch.cat([x, msg], dim=1)))
        out = w1 * msg + w2 * x
        
        return out

    def message(self, x_i, x_j, edge_index, edge_attr):
        edge_input = torch.cat([x_i, x_j - x_i, edge_attr], dim=1)
        
        msg = F.relu(self.mlp1(edge_input))

        w1 = F.tanh(self.mlp5(x_i))
        w2 = F.tanh(self.mlp6(msg))
        w = self.mlp7(w1 * w2)
        w = softmax(w, edge_index[0])
        
        return msg * w

class MODEL_GNN(nn.Module):
    def __init__(self):
        super(MODEL_GNN, self).__init__()
        self.conv1 = MPL(1, 4)  
        self.conv2 = MPL(4, 8)
        self.conv3 = MPL(8, 8)
        self.conv4 = MPL(8, 16)
        
        self.lin1 = nn.Linear(24, 32)
        self.lin2 = nn.Linear(32, 16)
        self.lin3 = nn.Linear(16, 16)
        self.lin4 = nn.Linear(16, 1)

        self.global_att_pool1 = GlobalAttention(nn.Sequential(nn.Linear(8, 1)))
        self.global_att_pool2 = GlobalAttention(nn.Sequential(nn.Linear(16, 1)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index, data.edge_attr))
        x = F.relu(self.conv2(x, edge_index, data.edge_attr))
        x1 = self.global_att_pool1(x, batch)
        
        x = F.relu(self.conv3(x, edge_index, data.edge_attr))
        x = F.relu(self.conv4(x, edge_index, data.edge_attr))
        x2 = self.global_att_pool2(x, batch)
        
        x_out = torch.cat([x1, x2], dim=1)  # Concatenate pooled outputs
        x = F.relu(self.lin1(x_out))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x).squeeze(1)

        return x
