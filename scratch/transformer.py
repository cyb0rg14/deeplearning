import math
import torch
import torch.nn as nn


# create 'input embedding' class
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# crete 'positional encoding' class 
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply the sin to even positions and cosine to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # -> (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)

    
# creating 'layer normalization' class
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiply
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# creating 'feed forward' class
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
        
    def forward(self, x):
        # (1, seq_len, d_model) -> (1, seq_len, d_ff) -> (1, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# creating 'multihead attention' class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_k = d_model // n_heads

        # create weight matrices of size (d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (1, seq_len, d_model) -> (1, seq_len, d_model)
        key = self.w_k(k) # (1, seq_len, d_model) -> (1, seq_len, d_model)
        value = self.w_v(v) # (1, seq_len, d_model) -> (1, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)


query = torch.randn(8, 6, 128)
query = query.view(query.shape[0], query.shape[1], 4, query.shape[2]//4).transpose(1, 2)
query.shape










# testing input embedding class
output = InputEmbedding(vocab_size=100, d_model=128)
print(output)

# testing positional encoding
another_output = PositionalEncoding(seq_len=100, d_model=128, dropout=0.1)
print(another_output)