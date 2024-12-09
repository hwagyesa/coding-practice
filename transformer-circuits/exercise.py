#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import dataclasses
import math
import operator
import pickle
import random
import re
import time
import unicodedata
from functools import reduce
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from hypothesis import given
from hypothesis import strategies as st
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from nanogpt_ref import CausalSelfAttention, NanoGPTConfig


class MHSA(nn.Module):
    """Multi-Head Self Attention (causal). No caching."""

    def __init__(self, d: int, h: int, L: int):
        super().__init__()
        self.d = d
        self.h = h
        self.L = L
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        nn.init.kaiming_normal_(self.qkv.weight, nonlinearity="linear")
        self.projection = nn.Linear(d, d, bias=False)
        nn.init.kaiming_normal_(self.projection.weight, nonlinearity="linear")

    def forward(self, x, mode=-1, tied=False):
        if mode == 0:
            return self._forward_matmul(x)
        elif mode == 1:
            return self._forward_coord(x)
        elif mode == 2:
            return self._forward_qk_vo(x)
        else:
            return self._forward_einsum(x, tied=tied)

    def _forward_einsum(self, x, tied=False):
        device = x.device
        # x is: b x L x d
        q, k, v = self.qkv(x).split(self.d, dim=-1)  # b x L x 3d
        if tied:
            q, k, v = q, q, q
        Q = q.view(q.shape[:-1] + (self.h, self.d // self.h))
        K = k.view(k.shape[:-1] + (self.h, self.d // self.h))
        V = v.view(v.shape[:-1] + (self.h, self.d // self.h))  # b x L x h x (d/h)
        scores = torch.einsum("blhd,bmhd->bhlm", Q, K) / math.sqrt(self.d / self.h)
        L = x.shape[-2]
        mask = (
            torch.arange(L, device=device)[:, None]
            < torch.arange(L, device=device)[None, :]
        )
        scores += torch.where(mask, -torch.inf, 0)  # to zero out, put -inf!
        attention = nn.functional.softmax(scores, dim=-1)
        # attention = nn.functional.softmax(scores, dim=-1)
        self_attention = torch.einsum("bhml,blhd->bmhd", attention, V)
        concatenated = torch.cat(self_attention.unbind(dim=-2), -1)
        out = self.projection(concatenated)
        return out

    def _forward_matmul(self, x):
        # x is: b x L x d
        device = x.device
        q, k, v = self.qkv(x).split(self.d, dim=-1)  # b x L x 3d
        Q = q.view(q.shape[:-1] + (self.h, self.d // self.h)).transpose(-3, -2)
        K = k.view(k.shape[:-1] + (self.h, self.d // self.h)).transpose(-3, -2)
        V = v.view(v.shape[:-1] + (self.h, self.d // self.h)).transpose(
            -3, -2
        )  # b x L x h x (d/h)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d / self.h)
        L = x.shape[-2]
        mask = (
            torch.arange(L, device=device)[:, None]
            < torch.arange(L, device=device)[None, :]
        )
        scores += torch.where(mask, -torch.inf, 0)  # to zero out, put -inf!
        attention = nn.functional.softmax(scores, dim=-1)
        self_attention = attention @ V  # b x h x L x (d/h)
        concatenated = torch.cat(self_attention.unbind(dim=-3), -1)  # b x L x d
        out = self.projection(concatenated)
        return out

    def _forward_coord(self, x):
        device = x.device
        # Explicitly extract the QKV matrices.
        # Our code above is reshaping to ..., h x 3 x (d/h)
        # So the relevant parts of the qkv projections are a little delocalized
        qkv_weight = self.qkv.weight  # 3d x d
        qkv_weight_split = qkv_weight.view(
            3, self.h, self.d // self.h, self.d
        )  # 3 x h x d/h x d
        q_proj, k_proj, v_proj = qkv_weight_split.unbind(0)  # each is h x d/h x d
        # x is b x L x d
        Q, K, V = (
            x[:, None, ...] @ q_proj.transpose(-1, -2),
            x[:, None, ...] @ k_proj.transpose(-1, -2),
            x[:, None, ...] @ v_proj.transpose(-1, -2),
        )  # output is b x h x L x d/h
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d / self.h)
        L = x.shape[-2]
        mask = (
            torch.arange(L, device=device)[:, None]
            < torch.arange(L, device=device)[None, :]
        )
        scores += torch.where(mask, -torch.inf, 0)  # to zero out, put -inf!
        attention = nn.functional.softmax(scores, dim=-1)
        self_attention = attention @ V  # b x h x L x d/h
        proj_weight = self.projection.weight  # d x d
        # Extract column submatrices
        proj_weight_split = proj_weight.T.view(self.h, self.d // self.h, self.d)
        # perform lifting
        out_blocks = self_attention @ proj_weight_split  # b x h x L x d
        out = out_blocks.sum(dim=1)  # b x L x d
        return out

    def _forward_qk_vo(self, x):
        device = x.device
        # Explicitly extract the QKV matrices.
        # Also employ the QK and VO parameterization. (compose matmuls)
        qkv_weight = self.qkv.weight  # 3d x d
        qkv_weight_split = qkv_weight.view(
            3, self.h, self.d // self.h, self.d
        )  # 3 x h x d/h x d
        q_proj, k_proj, v_proj = qkv_weight_split.unbind(0)  # each is h x d/h x d
        qk_proj = q_proj.transpose(-1, -2) @ k_proj  # h x d x d
        proj_weight = self.projection.weight  # d x d
        # Extract column submatrices
        proj_weight_split = proj_weight.T.view(
            self.h, self.d // self.h, self.d
        )  # h x d/h x d
        ov_proj = proj_weight_split.transpose(-1, -2) @ v_proj  # h x d x d
        # x is b x L x d
        head_scores = (
            x[:, None, ...] @ qk_proj @ x[:, None, ...].transpose(-2, -1)
        ) / math.sqrt(self.d / self.h)  # b x h x L x L
        L = x.shape[-2]
        mask = (
            torch.arange(L, device=device)[:, None]
            < torch.arange(L, device=device)[None, :]
        )
        head_scores += torch.where(mask, -torch.inf, 0)  # to zero out, put -inf!
        attention = nn.functional.softmax(head_scores, dim=-1)
        self_attention = attention @ x[:, None, ...]  # b x h x L x d
        # perform lifting
        out_heads = self_attention @ ov_proj.transpose(-2, -1)  # b x h x L x d
        out = out_heads.sum(dim=1)  # b x L x d
        return out

    def get_qk_ov(self):
        qkv_weight = self.qkv.weight  # 3d x d
        qkv_weight_split = qkv_weight.view(
            3, self.h, self.d // self.h, self.d
        )  # 3 x h x d/h x d
        q_proj, k_proj, v_proj = qkv_weight_split.unbind(0)  # each is h x d/h x d
        qk_proj = q_proj.transpose(-1, -2) @ k_proj  # h x d x d
        proj_weight = self.projection.weight  # d x d
        # Extract column submatrices
        proj_weight_split = proj_weight.T.view(
            self.h, self.d // self.h, self.d
        )  # h x d/h x d
        ov_proj = proj_weight_split.transpose(-1, -2) @ v_proj  # h x d x d
        return qk_proj, ov_proj


class MHSA0(nn.Module):
    def __init__(self, d, h, L):
        super().__init__()
        self.d = d
        self.h = h
        self.L = L

        # Parameters: qkv, out
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor):
        qkv = self.qkv(x)  # b x L x 3d
        qkv = qkv.view(
            x.shape[:-1] + (3, self.h, self.d // self.h)
        )  # b x L x 3 x h x (d/h)
        q, k, v = qkv.unbind(-3)  # each b x L x h x d/h
        logits = torch.einsum("blhd,bmhd->bhlm", q, k) / math.sqrt(
            self.d / self.h
        )  # b x h x L x L
        mask = torch.arange(x.shape[-2])[:, None] < torch.arange(x.shape[-2])[None, :]
        logits += torch.where(mask, -torch.inf, 0)
        attn = torch.softmax(logits, dim=-1)
        values = torch.einsum("bhml,blhd->bmhd", attn, v)  # b x L x h x d/h
        values = torch.cat(values.unbind(-2), dim=-1)
        out = self.proj(values)
        return out

    def get_qk_ov(self):
        # self.qkv.weight is 3d x d
        q_proj, k_proj, v_proj = self.qkv.weight.view(3, self.d, self.d).unbind(
            0
        )  # each d x d
        q_proj_h = q_proj.view(self.h, self.d // self.h, self.d)
        k_proj_h = k_proj.view(self.h, self.d // self.h, self.d)
        v_proj_h = v_proj.view(self.h, self.d // self.h, self.d)  # h x d/h x d
        o_proj_h = self.proj.weight.T.view(self.h, self.d // self.h, self.d)  # h x d/h x d
        # qk matrices
        qk = q_proj_h.transpose(-1, -2) @ k_proj_h
        ov = o_proj_h.transpose(-2, -1) @ v_proj_h

        return qk, ov


# EVAL:
# 1. only thing messed up was masking.
if __name__ == "__main__":
    b = 2
    h = 4
    d = 8
    L = 3
    k = 16
    atol = 1e-5

    data = torch.randn((b, L, d))
    ref_model = MHSA(d, h, L)
    model = MHSA0(d, h, L)
    model.qkv.weight = ref_model.qkv.weight
    model.proj.weight = ref_model.projection.weight

    # Test forward
    assert torch.allclose(ref_model(data), model(data), atol=atol)

    # Test qkov
    ref_qk, ref_ov = ref_model.get_qk_ov()
    qk, ov = model.get_qk_ov()

    assert torch.allclose(ref_qk, qk, atol=atol)
    assert torch.allclose(ref_ov, ov, atol=atol)
