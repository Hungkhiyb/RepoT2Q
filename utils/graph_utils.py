import torch
from torch_geometric.data import Data

def build_graph(vertices: torch.Tensor, faces: torch.Tensor, edge_features: torch.Tensor) -> Data:
    """Convert mesh to PyG graph."""
    edge_index = []
    # Tạo edges từ faces tam giác
    for face in faces:
        edge_index.append([face[0], face[1]])
        edge_index.append([face[1], face[2]])
        edge_index.append([face[2], face[0]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(
        x=vertices,            # Node features: [n_vertices, 3]
        edge_index=edge_index,  # Edge connections: [2, n_edges]
        edge_attr=edge_features # Edge features: [n_edges, 2]
    )