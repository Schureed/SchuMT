import torch.nn as nn

from schumt.model.module.attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_filter, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_head, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_filter),
            nn.ReLU(),
            nn.Linear(d_filter, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        src = self.norm1(src + self.dropout1(self.self_attention(src, src, src, mask)))
        src = self.norm2(src + self.dropout2(self.feed_forward(src)))

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_filter, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_head, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.encoder_decoder_attention = MultiHeadAttention(n_head, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_filter),
            nn.ReLU(),
            nn.Linear(d_filter, d_model),
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, trg, mem, target_mask=None, memory_mask=None):
        trg2 = self.dropout1(self.self_attention(trg, trg, trg, target_mask))
        trg = self.norm1(trg + trg2)

        trg2 = self.dropout2(self.encoder_decoder_attention(trg, mem, mem, memory_mask))
        trg = self.norm2(trg + trg2)

        trg2 = self.dropout3(self.feed_forward(trg))
        trg = self.norm3(trg + trg2)

        return trg


class TransformerCore(nn.Module):
    def __init__(self,
                 n_head,
                 num_encoder_layers,
                 num_decoder_layers,
                 d_model,
                 d_filter,
                 dropout,
                 **kwargs):
        super(TransformerCore, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(n_head, d_model, d_filter, dropout)] * num_encoder_layers)
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(n_head, d_model, d_filter, dropout)] * num_decoder_layers)

    def forward(self, src, trg, mask=None):
        for layer in self.encoder_layers:
            src = layer(src)
        for layer in self.decoder_layers:
            trg = layer(trg, src, mask)
        return trg


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, dropout, **params):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.transformer_arch = TransformerCore(d_model=d_model, dropout=dropout, **params)
        self.classifier = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, mask=None):
        src = self.dropout1(self.src_embedding(src))
        trg = self.dropout2(self.trg_embedding(trg))
        trg = self.transformer_arch(src, trg, mask)
        return self.classifier(trg)
