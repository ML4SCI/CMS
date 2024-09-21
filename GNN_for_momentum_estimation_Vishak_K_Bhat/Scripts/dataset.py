import torch
import itertools
from torch.utils.data import Dataset
from torch_geometric.data import Data

def connect_edges(sin_phi1, sin_phi2, cos_phi1, cos_phi2, eta1, eta2):
    return [sin_phi2 - sin_phi1, cos_phi2 - cos_phi1, eta2 - eta1]

class GraphDataset(Dataset):
    def __init__(self, dataset, labels, edge_index, indices, node_feat, transform=None):
        self.dataset = dataset
        self.labels = labels
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        self.indices = indices
        self.transform = transform
        self.node_feat = node_feat
    
    def __getitem__(self, idx):
        return self.convert_to_graph(self.indices[idx])
    
    def __len__(self):
        return len(self.indices)
    
    def convert_to_graph(self, i):
        row = self.dataset.iloc[i]
        sin_phi = torch.tensor(row[['sin_Phi_0', 'sin_Phi_2', 'sin_Phi_3', 'sin_Phi_4']].values, dtype=torch.float)
        cos_phi = torch.tensor(row[['cos_Phi_0', 'cos_Phi_2', 'cos_Phi_3', 'cos_Phi_4']].values, dtype=torch.float)
        eta = torch.tensor(row[['Eta_0', 'Eta_2', 'Eta_3', 'Eta_4']].values, dtype=torch.float)
        bend_angles = torch.tensor(row[['BendingAngle_0', 'BendingAngle_2', 'BendingAngle_3', 'BendingAngle_4']].values, dtype=torch.float)
        
        if(self.node_feat == "bendAngle"):
            node_features = bend_angles.unsqueeze(1)
        elif(self.node_feat == "etaValue"):
            node_features = eta.unsqueeze(1)
        edge_features_BA = []
        edge_features_EV = []

        num_nodes = len(bend_angles)
        
        for k, j in itertools.permutations(range(num_nodes), 2):
            sin_phi1, cos_phi1, eta1, bendAngle1 = sin_phi[k].item(), cos_phi[k].item(), eta[k].item(), bend_angles[k].item()
            sin_phi2, cos_phi2, eta2, bendAngle2 = sin_phi[j].item(), cos_phi[j].item(), eta[j].item(), bend_angles[j].item()
            edge_features_BA.append(connect_edges(sin_phi1, sin_phi2, cos_phi1, cos_phi2, eta1, eta2))
            edge_features_EV.append(connect_edges(sin_phi1, sin_phi2, cos_phi1, cos_phi2, bendAngle1, bendAngle2))

        
        edge_features_BA = torch.tensor(edge_features_BA, dtype=torch.float)
        edge_features_EV = torch.tensor(edge_features_EV, dtype=torch.float)

        if(self.node_feat == "bendAngle"):
            edge_features = edge_features_BA
        elif(self.node_feat == "etaValue"):
            edge_features = edge_features_EV


        data = Data(x=node_features, edge_index=self.edge_index, edge_attr=edge_features, y=torch.tensor(self.labels[i], dtype=torch.float))
        
        return data
