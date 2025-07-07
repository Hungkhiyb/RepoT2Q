import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, EdgeConv

class HybridEncoder(nn.Module):
    def __init__(self, node_dim=3, edge_dim=2, hidden_dim=64, heads=4):
        super().__init__()
        # EdgeConv cần input_dim = node_dim * 2 (x_i và x_j - x_i)
        self.edge_conv = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(node_dim * 2, hidden_dim),  # CHỈ dùng x_i và (x_j - x_i)
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        # Phần GAT riêng để xử lý edge features
        self.edge_processor = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.gat = GATConv(hidden_dim * 2, hidden_dim, heads=heads)  # Gộp cả node và edge features

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Bước 1: Xử lý hình học bằng EdgeConv (không dùng edge_attr)
        x_node = self.edge_conv(x, edge_index)
        
        # Bước 2: Xử lý edge features riêng
        x_edge = self.edge_processor(edge_attr)
        
        # Bước 3: Gộp thông tin
        row, col = edge_index
        x_combined = torch.cat([x_node[row], x_edge], dim=1)
        
        # Bước 4: GAT
        x = self.gat(x_combined, edge_index)
        return x