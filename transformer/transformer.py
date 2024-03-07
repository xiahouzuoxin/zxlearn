import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import InputEmbedding

class Attention(nn.Module):
    def __init__(self, d_model, d_k=64, bias=False):
        super(Attention, self).__init__()
        self.d_k = d_k
        self.Q_fc = nn.Linear(d_model, d_k, bias=bias)
        self.K_fc = nn.Linear(d_model, d_k, bias=bias)
        self.V_fc = nn.Linear(d_model, d_model, bias=bias)
        
    def forward(self, input):
        # input: (batch_size, seq_len, d_model)
        # output: (batch_size, seq_len, d_model)
        # softmax(QK^T / sqrt(d_k)) * V
        Q = self.Q_fc(input) # (batch_size, seq_len, d_k)
        K = self.K_fc(input) # (batch_size, seq_len, d_k)
        V = self.V_fc(input) # (batch_size, seq_len, d_model)
        attention = torch.bmm(Q, K.permute(0, 2, 1)) / (self.d_k ** 0.5)
        attention = F.softmax(attention, dim=2)
        output = torch.bmm(attention, V)
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, bias=False, dropout=0.1, qkv_proj=False):
        '''
        The original paper uses d_k = d_v = d_model / num_heads = 512 / 8 = 64
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0 and self.d_k > 0
        self.num_heads = num_heads

        self.qkv_proj = qkv_proj
        if qkv_proj:
            self.q_proj = nn.Linear(d_model, d_model, bias=bias)
            self.k_proj = nn.Linear(d_model, d_model, bias=bias)
            self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.output_fc = nn.Linear(d_model, d_model, bias=bias)
        
    def forward(self, Q, K, V, mask=None):
        # input: (batch_size, seq_len, d_model)
        # softmax(QK^T / sqrt(d_k)) * V

        B, T, d_model = Q.size()

        if self.qkv_proj:
            Q = self.q_proj(Q)
            K = self.k_proj(K)
            V = self.v_proj(V)

        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.d_k).permute(0, 2, 1, 3) # bs x num_heads x seq_len x d_k
        K = K.view(K.size(0), K.size(1), self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.d_k).permute(0, 2, 1, 3) # bs x num_heads x seq_len x d_k
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.d_k ** 0.5) # bs x num_heads x seq_len x seq_len
        if mask is not None:
            attention = attention.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1) # bs x num_heads x seq_len x seq_len
        attention = self.dropout(attention)
        output = torch.matmul(attention, V) # bs x num_heads x seq_len x d_k
        output = output.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        output = self.output_fc(output)
        return output
    
class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim, output_dim, bias=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        
    def forward(self, input):
        # input: (batch_size, seq_len, d_model)
        # output: (batch_size, seq_len, output_dim)
        output = F.relu(self.fc1(input))
        output = self.fc2(output)
        return output


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, bias=False, dropout=0.1):
        super(EncoderBlock, self).__init__()

        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=bias)

        self.attention = MultiHeadAttention(d_model, num_heads, bias=bias, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, bias=bias)

        self.mlp = MLP(d_model, d_model * 4, d_model, bias=bias) # in original paper, hidden_dim = 4 * d_model
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(d_model, bias=bias)
        
    def forward(self, input):
        # input: (batch_size, seq_len, d_model)
        # output: (batch_size, seq_len, d_model)

        QKV = self.qkv_proj(input)
        Q, K, V = torch.split(QKV, QKV.size(-1) // 3, dim=-1)

        attention = self.dropout1( self.attention(Q, K, V) )
        attention = attention + input
        attention = self.layer_norm1(attention)

        mlp = self.mlp(attention)
        output = self.dropout1(mlp)
        output = output + attention
        output = self.layer_norm2(output)
        return output

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, bias=False, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=bias)

        self.attention1 = MultiHeadAttention(d_model, num_heads, bias=bias, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, bias=bias)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=bias) # encoder output mapping to K and V

        self.attention2 = MultiHeadAttention(d_model, num_heads, bias=bias, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(d_model, bias=bias)

        self.mlp = MLP(d_model, d_model * 4, d_model, bias=bias) # in original paper, hidden_dim = 4 * d_model
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(d_model, bias=bias)

    def forward(self, input, enc_output, self_attn_mask=None):
        # input: (batch_size, seq_len, d_model)
        # output: (batch_size, seq_len, d_model)

        QKV = self.qkv_proj(input)
        Q, K, V = torch.split(QKV, QKV.size(-1) // 3, dim=-1)
        attention1 = self.dropout1( self.attention1(Q, K, V, self_attn_mask) )
        attention1 = attention1 + input
        attention1 = self.layer_norm1(attention1)

        Q = self.q_proj(attention1)
        KV = self.kv_proj(enc_output)
        K, V = torch.split(KV, KV.size(-1) // 2, dim=-1)
        attention2 = self.dropout2( self.attention2(Q, K, V) )
        attention2 = attention2 + attention1
        attention2 = self.layer_norm2(attention2)

        mlp = self.mlp(attention2)
        output = self.dropout3(mlp)
        output = output + attention2
        output = self.layer_norm3(output)
        return output
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, block_size, num_encoders=6, num_decoders=6, bias=False, dropout=0.1):
        '''
        vocab_size: the size of vocabulary
        d_model: the dimension of input and output
        num_heads: the number of heads in multi-head attention
        block_size: the max length of input and output sequence
        num_encoders: the number of encoder blocks
        num_decoders: the number of decoder blocks
        bias: whether to use bias in linear layers
        dropout: the dropout rate
        '''
        super(Transformer, self).__init__()
        self.input_embedding = InputEmbedding(vocab_size, d_model, max_len=block_size, dropout=dropout)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, bias, dropout) for _ in range(num_encoders)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, bias, dropout) for _ in range(num_decoders)])
        self.fc = nn.Linear(d_model, vocab_size, bias=bias)

        self.register_buffer('mask', 
                             torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
                             .view(1, 1, block_size, block_size)
        )

    def forward(self, src_input, tgt_input):
        # src_input: (batch_size, src_seq_len)
        # tgt_input: (batch_size, tgt_seq_len)
        # output: (batch_size, tgt_seq_len, vocab_size)
        src_output = self.input_embedding(src_input)
        tgt_output = self.input_embedding(tgt_input)
        for encoder_block in self.encoder_blocks:
            src_output = encoder_block(src_output)
        for decoder_block in self.decoder_blocks:
            tgt_output = decoder_block(tgt_output, src_output, self.mask)
        output = self.fc(tgt_output)
        return output
