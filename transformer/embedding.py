import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', self._get_sinusoid_encoding_table(d_model, max_len)) # don't need to update parameters
        
    def _get_sinusoid_encoding_table(self, d_model, max_len):
        import numpy as np
        def cal_angle(position, i):
            return position / np.power(10000, 2 * (i // 2) / d_model)
        def get_posi_angle_vec(position):
            return [cal_angle(position, i) for i in range(d_model)]
        sinusoid_table = torch.tensor([get_posi_angle_vec(pos) for pos in range(max_len)], dtype=torch.float)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2]) # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2]) # dim 2i+1
        return sinusoid_table.unsqueeze(0)
    
    def forward(self, seq_len):
        return self.pe[:, :seq_len]

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout=0.1):
        super(InputEmbedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        # input: (batch_size, seq_len)
        # output: (batch_size, seq_len, d_model)
        output = self.tok_embed(input) * (self.tok_embed.embedding_dim ** 0.5) # page 5 of the original paper
        output = output + self.pos_embed(input.size(1)).to(output.device)
        output = self.dropout(output)
        return output