import trimesh
import torch
import numpy as np
from typing import Tuple

def load_obj(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from .obj file."""
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices  # Shape: [n_vertices, 3]
    faces = mesh.faces        # Shape: [n_faces, 3] (triangles)
    return vertices, faces

def compute_edge_features(vertices, faces):
    edge_features = []
    edge_map = {}  # Ánh xạ (v0,v1) → index
    
    # Tạo danh sách cạnh duy nhất
    edges = []
    for face in faces:
        edges.extend([
            (face[0], face[1]),
            (face[1], face[2]), 
            (face[2], face[0])
        ])
    
    # Tính toán features cho từng cạnh
    for i, (v0, v1) in enumerate(edges):
        if (v0, v1) not in edge_map:
            # Tính vector và độ dài
            vec = vertices[v1] - vertices[v0]
            length = np.linalg.norm(vec)
            
            # Chuẩn hóa vector làm feature
            edge_features.append([length, 0.5])  # Có thể thêm các features khác
            edge_map[(v0, v1)] = len(edge_features) - 1
    
    return torch.tensor(edge_features, dtype=torch.float)