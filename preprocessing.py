import os
import yaml
import torch
from tqdm import tqdm  # Thêm thư viện hiển thị tiến trình
from utils.mesh_processing import load_obj, compute_edge_features
from utils.graph_utils import build_graph

def process_single_mesh(obj_path: str, output_dir: str) -> None:
    """Xử lý 1 file .obj và lưu đồ thị tương ứng."""
    try:
        # Tạo tên file output (đổi đuôi .obj → .pt)
        mesh_name = os.path.splitext(os.path.basename(obj_path))[0]
        output_path = os.path.join(output_dir, f"{mesh_name}_graph.pt")

        # Bỏ qua nếu file đã tồn tại
        if os.path.exists(output_path):
            return

        # Tiền xử lý
        vertices, faces = load_obj(obj_path)
        vertices = torch.tensor(vertices, dtype=torch.float)
        faces = torch.tensor(faces, dtype=torch.long)
        edge_features = compute_edge_features(vertices.numpy(), faces.numpy())

        # Xây dựng đồ thị và lưu
        graph_data = build_graph(vertices, faces, edge_features)
        torch.save(graph_data, output_path)

    except Exception as e:
        print(f"\nError processing {obj_path}: {e}")

def main():
    # Load config
    with open("configs/preprocessing.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Đảm bảo thư mục output tồn tại
    os.makedirs(config["output_dir"], exist_ok=True)

    # Lấy danh sách file .obj trong thư mục input
    obj_files = [
        os.path.join(config["input_dir"], f)
        for f in os.listdir(config["input_dir"])
        if f.endswith(".obj")
    ]

    print(f"Found {len(obj_files)} .obj files to process.")

    # Xử lý từng file với thanh tiến trình
    for obj_path in tqdm(obj_files, desc="Processing meshes", unit="mesh"):
        process_single_mesh(obj_path, config["output_dir"])

    print(f"\nDone! Processed graphs saved to {config['output_dir']}")

if __name__ == "__main__":
    main()