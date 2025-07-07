import os
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from models.hybrid_encoder import HybridEncoder
from models.transformer import MeshTransformer

# --- Dataset Class ---
class MeshGraphDataset(Dataset):
    def __init__(self, processed_dir):  # Đổi tên tham số thành processed_dir
        self.graph_files = [
            os.path.join(processed_dir, f) 
            for f in os.listdir(processed_dir) 
            if f.endswith(".pt")  # Chỉ đọc file .pt từ Bước 1
        ]

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        return torch.load(self.graph_files[idx], weights_only=False)

# --- Collate Function ---
def collate_fn(batch):
    return batch  # Giữ nguyên định dạng PyG Data

# --- Main Function ---
def main():
    # Load config
    with open("configs/encoding.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Khởi tạo model
    encoder = HybridEncoder(
        hidden_dim=config["model_params"]["hidden_dim"],
        heads=config["model_params"]["gat_heads"]
    )
    transformer = MeshTransformer(
        hidden_dim=config["model_params"]["hidden_dim"] * config["model_params"]["gat_heads"],
        nhead=config["model_params"]["transformer_heads"]
    )

    # Tạo dataset và dataloader
    dataset = MeshGraphDataset(config["processed_dir"])  
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_fn)

    # Xử lý từng batch
    os.makedirs(config["features_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    transformer.to(device)

    for batch in tqdm(dataloader, desc="Encoding graphs"):
        features_batch = []
        for data in batch:
            # Chuyển dữ liệu sang device
            data = data.to(device)
            # Extract features
            with torch.no_grad():
                node_features = encoder(data)
                graph_features = transformer(node_features.unsqueeze(0)).mean(dim=1)  # Global pooling
            features_batch.append(graph_features.cpu())

        # Lưu features
        for i, data in enumerate(batch):
            mesh_name = os.path.splitext(os.path.basename(dataset.graph_files[i]))[0]
            torch.save(features_batch[i], os.path.join(config["output_dir"], f"{mesh_name}_features.pt"))

if __name__ == "__main__":
    main()