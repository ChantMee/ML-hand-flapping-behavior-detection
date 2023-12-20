import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, attn_mask=mask, need_weights=False)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.output_linear(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src = src + self.dropout1(self.self_attn(src2, src2, src2, src_mask))
        src2 = self.norm2(src)
        src = src + self.dropout2(F.relu(self.linear1(src2)))
        src = src + self.dropout(self.linear2(src))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        return self.norm(src)


