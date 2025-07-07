import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MeshTransformer(nn.Module):
    def __init__(self, hidden_dim=64, nhead=8, num_layers=3):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, padding_mask=None):
        # x shape: [batch_size, num_nodes, hidden_dim]
        x = x.transpose(0, 1)  # Transformer cần [seq_len, batch, features]
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return x.transpose(0, 1)