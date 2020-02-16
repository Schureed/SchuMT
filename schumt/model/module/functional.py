import math

import torch


def batch_dot_product_attention(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask):
    """
    q: (H, N, T, E)
    k: (H, N, S, E)
    v: (H, N, S, E)
    assert q.dim() == k.dim() == v.dim() == 4
    assert q.size(0) == k.size(0) == v.size(0)
    assert q.size(1) == k.size(1) == v.size(1)
    assert k.size(2) == v.size(2)
    assert q.size(3) == k.size(3) == v.size(3)
    if mask is not None:
        assert mask.dim() == 2
        assert mask.size(0) == q.size(2)
        assert mask.size(1) == k.size(2)
    """

    d_head = q.size(3)
    scaling_factor = math.sqrt(d_head)
    attn = torch.matmul(q, k.transpose(2, 3)) / scaling_factor
    if mask is not None:
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def generate_target_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
