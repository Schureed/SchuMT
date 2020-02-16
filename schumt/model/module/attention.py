import torch
import torch.nn as nn

from schumt.model.module.functional import batch_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        if d_model % n_head == 0:
            self.d_head = d_model // n_head
        else:
            raise ValueError("h_head should divide d_model")

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wout = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        :param q: (N, T, E)
        :param k: (N, S, E)
        :param v: (N, S, E)
        :param mask: (T, T) or None
        :return: (N, T, E)
        """
        qs = torch.stack(self.wq(q).chunk(chunks=self.n_head, dim=-1))
        ks = torch.stack(self.wk(k).chunk(chunks=self.n_head, dim=-1))
        vs = torch.stack(self.wv(v).chunk(chunks=self.n_head, dim=-1))

        out = batch_dot_product_attention(qs, ks, vs, mask)
        out = torch.cat(out.unbind(dim=0), dim=-1)
        return self.wout(out)
