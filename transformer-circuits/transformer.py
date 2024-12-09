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


# TODO: Pack into a BPE tokenizer to study at larger scale? (tinystories is printable ascii... typical LM setting has k >> d by 1 oom...)
@dataclasses.dataclass
class PrintableASCIITokenizer:
    """Simple printable ASCII tokenizer"""

    # Printable ASCII chars have control codes from 32 to 126
    # That means our tokenizer has a vocab size of 95.

    def encode(self, text: str) -> list[int]:
        assert (
            text.isprintable()
        ), "Input string characters need to be valid printable ASCII"
        return [ord(c) - 32 for c in text]

    def decode(self, tokens: list[int]) -> str:
        assert all(
            0 <= i <= 94 for i in tokens
        ), "Input tokens need to correspond to valid printable ASCII"
        return "".join(chr(32 + i) for i in tokens)


@dataclasses.dataclass
class ASCIITokenizer:
    """Simple ASCII tokenizer, vocab size 128"""

    def encode(self, text: str) -> list[int]:
        assert text.isascii(), "Input string characters need to be valid ASCII"
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        assert all(
            0 <= i <= 127 for i in tokens
        ), "Input tokens need to correspond to valid ASCII"
        return "".join(chr(i) for i in tokens)


@dataclasses.dataclass
class ASCIITokenizerSpecial:
    """Simple ASCII tokenizer, vocab size 129 (1 special token)"""

    def encode(self, text: str) -> list[int]:
        assert text.isascii(), "Input string characters need to be valid ASCII"
        # Process special character "<|endoftext|>"
        pat = re.compile("(" + re.escape("<|endoftext|>") + ")")
        special_split = re.split(pat, text)
        tokens = []
        for string in special_split:
            if string == "<|endoftext|>":
                tokens += [128]
            else:
                tokens += [ord(c) for c in string]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        assert all(
            0 <= i <= 128 for i in tokens
        ), "Input tokens need to correspond to valid ASCII + 1 special"
        # Scan for special tokens. Lazily convert them to Unicode replacement char
        codepoint = 65533
        for i, tok in enumerate(tokens):
            if tok == 128:
                tokens[i] = codepoint
        return "".join(chr(i) for i in tokens)


# TODO: We aren't actually using the context length anywhere. Technically we should
#  "sliding window" the self-attention mask for cases where we pass a very long input...
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


def test_mhsa():
    b = 16
    L = 1024
    h = 12
    d = 768
    atol = 1e-5
    # b = 2
    # L = 10
    # h = 2
    # d = 8
    data = torch.randn((b, L, d))
    model = MHSA(d, h, L)
    ref_config = NanoGPTConfig(
        n_embd=d, n_head=h, bias=False, dropout=0.0, block_size=L
    )
    ref_model = CausalSelfAttention(ref_config)
    ref_model.c_attn.weight = model.qkv.weight
    ref_model.c_proj.weight = model.projection.weight
    # our_q = model.qkv.weight[:d, :]
    # our_k = model.qkv.weight[d:2*d, :]
    # our_v = model.qkv.weight[2*d:, :]
    # our_o = model.projection.weight
    out0 = model(data, mode=0)
    out1 = model(data, mode=1)
    out2 = model(data, mode=2)
    out3 = model(data)
    out4 = ref_model(data)
    # test_model = nn.MultiheadAttention(d, h, bias=False, batch_first=True)
    # test_model.in_proj_weight.data = model.qkv.weight
    # test_model.out_proj.weight = model.projection.weight
    # test_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    # out4 = test_model(data, data, data, need_weights=False)
    # out4, out4_attn = test_model(data, data, data, attn_mask=test_mask, is_causal=True)
    assert torch.all(torch.isclose(out0, out1, atol=atol))
    assert torch.all(torch.isclose(out1, out2, atol=atol))
    assert torch.all(torch.isclose(out2, out3, atol=atol))
    assert torch.all(torch.isclose(out3, out4, atol=atol))

    # Benchmarking
    runs = 10
    t0 = []
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    for run in tqdm(range(runs)):
        tstart = time.perf_counter()
        data = torch.randn((b, L, d))
        out0 = model(data, mode=0)
        t0.append(time.perf_counter() - tstart)

        tstart = time.perf_counter()
        data = torch.randn((b, L, d))
        out1 = model(data, mode=1)
        t1.append(time.perf_counter() - tstart)

        tstart = time.perf_counter()
        data = torch.randn((b, L, d))
        out2 = model(data, mode=2)
        t2.append(time.perf_counter() - tstart)

        tstart = time.perf_counter()
        data = torch.randn((b, L, d))
        out3 = model(data, mode=-1)
        t3.append(time.perf_counter() - tstart)

        tstart = time.perf_counter()
        data = torch.randn((b, L, d))
        out4 = model(data, mode=-1)
        t4.append(time.perf_counter() - tstart)

    print(f"einsum mode: mean {mean(t3)}, stdev {stdev(t3)}")
    print(f"matmul mode: mean {mean(t0)}, stdev {stdev(t0)}")
    print(f"manual weights mode: mean {mean(t1)}, stdev {stdev(t1)}")
    print(f"qkvo mode: mean {mean(t2)}, stdev {stdev(t2)}")
    print(f"nanogpt ref model: mean {mean(t2)}, stdev {stdev(t2)}")

    # time.sleep(10000000)


# (input stage: need to pad. output stage: need to enforce correct loss calculation.)
class TransformerAttnOnly(nn.Module):
    """Transformer architecture, Attention-Only version!"""

    def __init__(
        self,
        k: int,  # vocab size
        d: int,  # embedding dim
        h: int,  # num heads
        layers: int,  # num layers
        context_length: int,  # context length
        crate: bool = False,  # tied-style attention
    ):
        super().__init__()
        # general
        self.tied = crate
        # Embedding stage
        self.embedding: nn.Embedding = nn.Embedding(k, d)
        self.context_length = context_length
        self.embedding_pos = nn.Parameter(
            torch.randn((context_length, d)), requires_grad=True
        )
        # Blocks: layers
        transformer_blocks: list[MHSA] = []
        for layer in range(layers):
            attn = MHSA(d=d, h=h, L=context_length)
            transformer_blocks.append(attn)
            # Register parameters for non-Sequential construction
            self.add_module(f"MHSA {layer}", attn)
        # self.transformer = nn.Sequential(*transformer_blocks)
        self.transformer_blocks = transformer_blocks
        # Output stage
        # self.output_layernorm = nn.LayerNorm(d)
        # self.unembedding = nn.Parameter(torch.randn(d, k))
        self.unembedding = nn.Linear(d, k, bias=False)
        nn.init.kaiming_normal_(self.unembedding.weight, nonlinearity="linear")

    def forward(self, x):
        x = self._embed(x)
        for attn in self.transformer_blocks:
            x = x + attn(x, tied=self.tied)
        x = self._unembed(x)
        return x

    def _embed(self, x):
        # return self.embedding(x) + self.embedding_pos[None, : x.shape[1], :]
        return self.embedding(x)

    def _unembed(self, x):
        # x = self.output_layernorm(x)
        # logits = x @ self.embedding.weight.T  # b x L x k
        # logits = x @ self.unembedding  # b x L x k
        logits = self.unembedding(x)
        return logits

    def get_qk_ov_circuits(self):
        E = self.embedding.weight.T  # d x k
        U = self.unembedding.weight  # k x d
        qkov_ckt = []
        for block in self.transformer_blocks:
            qk, ov = block.get_qk_ov()  # each is d x d
            qkov_ckt.append((E.T @ qk @ E, U @ ov @ E))
        direct = U @ E
        return qkov_ckt, direct


def test_qkov():
    h = 4
    d = 8
    L = 2
    k = 16
    atol = 1e-5
    # model = MHSA(d, h, L)
    # qk, ov = model.get_qk_ov()
    # model = TransformerAttnOnly(k, d, h, 1, L)

    load_path = "model_1layer_attnonly.pth"
    # Load from file
    model_info = torch.load(load_path)
    model = TransformerAttnOnly(**model_info["params"])
    model.load_state_dict(model_info["state_dict"])

    # model.to('cuda')
    qkov_ckt, direct = model.get_qk_ov_circuits()
    # model.to('cpu')
    plt.imshow(direct.detach().cpu())
    plt.title("direct path matrix")
    plt.show()
    for qk_h, ov_h in qkov_ckt:
        for h, qk in enumerate(qk_h.unbind(0)):
            plt.imshow(qk.detach().cpu())
            plt.title(f"qk circuit, head {h}")
            plt.show()
        for h, ov in enumerate(ov_h.unbind(0)):
            plt.imshow(ov.detach().cpu())
            plt.title(f"ov circuit, head {h}")
            plt.show()

    print("a")


@given(st.lists(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126))))
def test_tokenizer(texts: list[str]):
    tok = PrintableASCIITokenizer()
    assert all(
        map(operator.eq, texts, (tok.decode(tok.encode(text)) for text in texts))
    )


def tokenize_and_pad(texts: list[str]) -> tuple[list[list[int]], list[int]]:
    """Input batch of document, output tokenized documents (w/ padding)"""
    pad_token = 128
    tokenizer = ASCIITokenizer()
    tokens_list = [tokenizer.encode(text) for text in texts]
    first_pad_token_idx = list(map(len, tokens_list))
    L = max(first_pad_token_idx)
    padded_tokens_list = [
        tokens + [pad_token] * (L - i)
        for (tokens, i) in zip(tokens_list, first_pad_token_idx)
    ]
    return padded_tokens_list, first_pad_token_idx


def normalize_quotes(text):
    """Scrub unicode quotes from tinystories"""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def prepare_tinystories(path: Path = Path("./data"), max_context_length: int = 1024):
    """Get a training set of documents. Filter by our context length."""
    train_path = path / Path("TinyStoriesV2-GPT4-train.txt")
    val_path = path / Path("TinyStoriesV2-GPT4-valid.txt")
    # Split pattern
    pat = re.compile(re.escape("<|endoftext|>"))

    with open(val_path, "r+") as f:
        text = f.read()
        documents = re.split(pat, text)
    valid_documents = [
        normalize_quotes(doc) for doc in documents if len(doc) <= max_context_length
    ]
    tokens_list, first_pad_idx = tokenize_and_pad(valid_documents)
    return valid_documents, tokens_list, first_pad_idx


def prepare_tinystories_block(
    path: Path = Path("./data"), max_context_length: int = 1024
):
    """Get a training set of documents; concatenate into a big corpus that we'll subsample"""
    train_path = path / Path("TinyStoriesV2-GPT4-train.txt")
    val_path = path / Path("TinyStoriesV2-GPT4-valid.txt")

    tokenizer = ASCIITokenizerSpecial()
    with open(train_path, "r+") as f:
        text = normalize_quotes(f.read())

    return tokenizer.encode(text)


def get_unigram_bigram_freqs(tokens, chunk_size=1024 * 1024):
    unigram_freqs = torch.zeros((num_vocab,))
    for i in tqdm(range(0, len(tokens), chunk_size)):
        batch = tokens[i : i + chunk_size]
        counts = torch.bincount(batch, minlength=num_vocab)
        unigram_freqs += counts
    bigram_freqs = torch.zeros((num_vocab, num_vocab))
    for i in tqdm(range(0, len(tokens) - 1, chunk_size)):
        batch = tokens[:-1][i : i + chunk_size].to(torch.int64)
        batch_next = tokens[i + 1 : i + 1 + chunk_size].to(torch.int64)
        bigram_freqs.index_put_(
            (batch, batch_next), torch.ones(len(batch)), accumulate=True
        )
    return unigram_freqs, bigram_freqs


def sample(temperature: float, logits: torch.Tensor):
    if temperature > 0:
        probs = nn.functional.softmax(logits / temperature, dim=0)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = logits.argmax(keepdim=True)
    return next_token


if __name__ == "__main__":
    num_vocab_base = 128
    num_vocab_special = 1  # pad token
    num_vocab = num_vocab_base + num_vocab_special
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # # Process data from scratch + dump tokens
    # tokens_list = prepare_tinystories_block(
    #     max_context_length=context_length + 1  # add 1 for ntp setup
    # )
    # tokens = torch.tensor(tokens_list, dtype=torch.uint8)
    # with open('tokens.pkl', 'wb') as f:
    #     pickle.dump(tokens, f)
    # Load preprocessed data
    print("Loading dataset of tokens.")
    with open("tokens.pkl", "rb") as f:
        tokens = pickle.load(f)
    print("Data loaded.")
    # Some preprocessing for 0-layer experiments: calculate unigram and bigram stats
    # chunk_size = 1024 * 1024
    # unigram_freqs = torch.zeros((num_vocab,))
    # for i in tqdm(range(0, len(tokens), chunk_size)):
    #     batch = tokens[i : i + chunk_size]
    #     # Use torch.bincount for efficient counting
    #     counts = torch.bincount(batch, minlength=num_vocab)
    #     unigram_freqs += counts
    # bigram_freqs = torch.zeros((num_vocab, num_vocab))
    # for i in tqdm(range(0, len(tokens) - 1, chunk_size)):
    #     batch = tokens[:-1][i : i + chunk_size].to(torch.int64)
    #     batch_next = tokens[i + 1 : i + 1 + chunk_size].to(torch.int64)
    #     bigram_freqs.index_put_((batch, batch_next), torch.ones(len(batch)), accumulate=True)
    # with open("unigrams.pkl", "wb") as f:
    #     pickle.dump(unigram_freqs, f)
    # with open("bigrams.pkl", "wb") as f:
    #     pickle.dump(bigram_freqs, f)
    with open("unigrams.pkl", "rb") as f:
        unigram_freqs = pickle.load(f)
    with open("bigrams.pkl", "rb") as f:
        bigram_freqs = pickle.load(f)

    context_length = 256
    load_model = False
    save_path = "model_3layer_attnonly.pth"
    load_path = "model_6layer_attnonly.pth"
    model_params = {
        "k": num_vocab,
        "d": 384,
        "h": 6,
        "layers": 3,
        "context_length": context_length,
        "crate": False,
    }
    model = TransformerAttnOnly(**model_params)
    model.to(device)

    # Hyperparameters
    batches = 4 * 1024
    batch_size = 64
    lr = 1e-3
    wd = 1e-1

    # Prepare data
    # ?: How do we load data for a next-token-prediction training...?
    # In nanogpt, Karpathy does simple 'sampling with replacement', rather than a typical shuffling
    # This is likely quite reasonable for massive text corpus training, where we typically don't
    # even train 1 full epoch over the training data! We don't expect too many collisions; and
    # shuffling the data would be extremely costly (c.f. the Anthropic engineering talk...)
    num_tokens = tokens.shape[0]

    # Prepare optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=wd)
    ce_loss = torch.nn.CrossEntropyLoss()

    if not load_model:
        # Train
        model.train()
        pbar = tqdm(range(batches))
        for batch in pbar:
            # Get a batch of data, sampling with replacement
            idxs = torch.randint(num_tokens - context_length, (batch_size,))
            x_batch = torch.stack([tokens[i : i + context_length] for i in idxs])
            y_batch = torch.stack(
                [tokens[i + 1 : i + 1 + context_length] for i in idxs]
            )
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch.to(torch.int32))
            loss = ce_loss(logits.transpose(-2, -1), y_batch.to(torch.int64))
            # CE loss is reduced with 'mean' by default + this takes a mean over the sequence dim too
            # So these loss values are 'optimally' of size about log(batch_size) / context_length

            loss.backward()
            # TODO: unstable training even with grad clipping sometimes at 6layer model. Find out why
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({"batch loss": f"{loss.detach():.3f}"})
        # For zero-layer transformer, the following is NOT the right optimal loss
        # Optimal loss would actually be
        #   E_{x,y ~ p(x,y)}[-log p(y|x)] = H(y|x) = H(x, y) - H(x)
        # So it's *always less* than the joint entropy,
        #   the optimal loss if we were really doing bigram modeling
        # The same caveat is valid for general autoregressive modeling:
        #   we're always conditional on first token
        # but for general autoregressive modeling, y becomes "rest of the sequence",
        #   so the effect on optimal loss is diminished
        p = bigram_freqs.reshape(-1) / bigram_freqs.sum()
        q = unigram_freqs.reshape(-1) / unigram_freqs.sum()
        print(f"empirical bigram entropy: {-p @ torch.log(p + 1e-8)}")
        print(f"empirical unigram entropy: {-q @ torch.log(q + 1e-8)}")
        print(
            f"empirical conditional entropy: {-p @ torch.log(p + 1e-8) + q @ torch.log(q + 1e-8)}"
        )
        # Save model
        model_info = {
            "state_dict": model.to("cpu").state_dict(),
            "params": model_params,
        }
        torch.save(model_info, save_path)

    else:
        # Load from file
        model_info = torch.load(load_path)
        model = TransformerAttnOnly(**model_info["params"])
        model.load_state_dict(model_info["state_dict"])

    # Sample
    model.to(device)
    model.eval()
    # Get some prompts
    docs, _, __ = prepare_tinystories(max_context_length=512)
    i = random.randint(0, len(docs) - 1)
    prompt = docs[i][: context_length // 2]
    print("The prompt: " + prompt)
    # inference: temperature-based
    temperature = 0.5
    # TODO: KV-caching
    # TODO: batch inference
    # TODO: More fun sampling strategies
    num_toks = context_length // 2
    tokenizer = ASCIITokenizerSpecial()
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.int32).to(device)
    with torch.no_grad():
        gen = ""
        for gens in tqdm(range(num_toks)):
            logits = model(tokens.unsqueeze(0))[0, -1, ...]
            next_token = sample(temperature, logits)
            tokens = torch.cat((tokens, next_token))
            gen += tokenizer.decode([int(next_token)])
    print("The completion: " + gen)

    # TODO: look at some small model scaling laws (calc compute FLOPS, chinchilla-style)
    time.sleep(1000)
