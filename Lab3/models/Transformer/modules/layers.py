import torch.nn as nn
import torch
import math


# TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.N_heads = num_heads
        self.d_k = dim // num_heads
        self.d_v = dim // num_heads
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def _scaled_dot_product_attention(self, Q, K, V):
        # Q, K, V shape: (B, N_heads=16, N=256, 48)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)
        output = torch.matmul(attn, V)
        return output

    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        B = x.size(0)
        N = x.size(1)

        # Reshape from (B, N=256, 768) → (B, N=256, 16, 48) → (B, 16, N=256, 48)
        Q = Q.view(B, N, self.N_heads, self.d_k).transpose(1, 2)
        K = K.view(B, N, self.N_heads, self.d_k).transpose(1, 2)
        V = V.view(B, N, self.N_heads, self.d_v).transpose(1, 2)

        attn_output = self._scaled_dot_product_attention(Q, K, V) # (B, 16, N=256, 48)
        output = self.W_out(
            attn_output.transpose(1, 2).contiguous().view(B, N, -1) # (B, N=256, 768)
        )
        return output


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=drop_rate),
        )

    def forward(self, input):
        return super().forward(input)


class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12),
        )

    def forward(self, input):
        return super().forward(input)


class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)

        x = x + attn
        x = self.LayerNorm1(x)

        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
