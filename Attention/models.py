import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None, mode='soft'):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    if mode == 'soft':
        attention = F.softmax(attn_logits, dim=-1)
    elif mode == 'hard':
        attention = torch.zeros_like(attn_logits)
        indices = torch.argmax(attn_logits, dim=-1).unsqueeze(dim=-1)
        attention = attention.scatter(-1, indices, 1.0)

    values = torch.matmul(attention, v)
    return values


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, mode='soft'):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.mode = mode

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, input_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        values = scaled_dot_product(q, k, v, mask=mask, mode=self.mode)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        values = self.o_proj(values)

        return values


class Encoder(nn.Module):
    def __init__(self,
                 input_dim=3,
                 embed_dim=9,
                 linear_dim=9,
                 dropout=0.0,
                 n_heads=3,
                 mode='soft'):
        super(Encoder, self).__init__()
        self.attention = MultiheadAttention(input_dim=input_dim,
                                            embed_dim=embed_dim,
                                            n_heads=n_heads,
                                            mode=mode)

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.linear = nn.Sequential(
            nn.Linear(input_dim, linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layer_norm1(x)
        attn_out = self.attention(x)
        attn_out = x + attn_out

        attn_out = self.layer_norm2(attn_out)
        linear_out = self.linear(attn_out)
        out = attn_out + linear_out
        return out


class SelfAttentionModel(nn.Module):
    def __init__(self, n_layers, input_dim, embed_dim, linear_dim, dropout, n_heads, mode):
        super().__init__()
        self.layers = nn.ModuleList([Encoder(input_dim=input_dim,
                                             embed_dim=embed_dim,
                                             linear_dim=linear_dim,
                                             dropout=dropout,
                                             n_heads=n_heads,
                                             mode=mode) for _ in range(n_layers)])
        self.mlp = nn.Linear(input_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        out, _ = x.max(dim=1)
        out = self.mlp(out)

        return out

