#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import hydra
import torch
import torchvision.transforms as transforms
from einops import einsum, rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import wandb


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int):
        super().__init__()
        assert input_dim % num_heads == 0
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.qkv = nn.Linear(input_dim, 3 * input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.projection = nn.Linear(input_dim, input_dim)
        self.scale = (self.input_dim // self.num_heads) ** (-0.5)

    def forward(self, input, debug=False):
        x = self.norm(input)  # x is b x n x d
        qkv = self.qkv(x)  # this is b x n x 3d
        # if debug:
        #     q = self.qkv.weight[0 : self.input_dim, :]
        #     k = self.qkv.weight[self.input_dim : 2 * self.input_dim, :]
        #     v = self.qkv.weight[2 * self.input_dim :, :]
        #     assert torch.all(qkv[0, :, 0 : self.input_dim].T == q @ x[0, ...].T)
        #     assert torch.all(
        #         qkv[0, :, self.input_dim : 2 * self.input_dim].T == k @ x[0, ...].T
        #     )
        #     assert torch.all(qkv[0, :, 2 * self.input_dim :].T == v @ x[0, ...].T)
        # Head reshaping. This is now b x h x 3 x n x d/h
        qkv_heads = rearrange(qkv, "b n (c h d) -> b h c n d", c=3, h=self.num_heads)
        attn = self.softmax(
            self.scale
            * einsum(
                qkv_heads[:, :, 0, ...],
                qkv_heads[:, :, 1, ...],
                "b h n1 d, b h n2 d -> b h n1 n2",
            )
        )
        # if debug:
        #     q_proj = (q @ x[0, ...].T).T
        #     q_proj_h0 = q_proj[:, : self.input_dim // self.num_heads]
        #     k_proj = (k @ x[0, ...].T).T
        #     k_proj_h0 = k_proj[:, : self.input_dim // self.num_heads]
        #     qk_h0 = q_proj_h0 @ k_proj_h0.T
        #     softmax_h0 = torch.nn.functional.softmax(qk_h0, dim=-1)
        #     assert torch.all(softmax_h0 == attn[0, 0, ...])
        self_attn = einsum(
            qkv_heads[:, :, 2, ...], attn, "b h n2 d, b h n1 n2 -> b h n1 d"
        )
        # if debug:
        #     v_proj = (v @ x[0, ...].T).T
        #     v_proj_h0 = v_proj[:, : self.input_dim // self.num_heads]
        #     self_attn_h0 = (v_proj_h0.T @ softmax_h0.T).T
        #     assert torch.all(self_attn_h0 == self_attn[0, 0, ...])
        projected = self.projection(rearrange(self_attn, "b h n d -> b n (h d)"))
        return projected


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, rescale_factor: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.embed = nn.Linear(input_dim, rescale_factor * input_dim)
        self.activation = nn.GELU(approximate="tanh")
        self.readout = nn.Linear(rescale_factor * input_dim, input_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.embed(x)
        x = self.activation(x)
        x = self.readout(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        ffnn_rescale_factor: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.sequence_length = (self.input_dim // self.patch_size) ** 2
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffnn_rescale_factor = ffnn_rescale_factor
        self.patch_embedding_layernorm = nn.LayerNorm(self.embedding_dim)
        self.num_classes = num_classes

        assert input_dim % patch_size == 0

        # Embed with non-overlapping patches.
        self.embedding = nn.Linear(
            3 * self.patch_size**2, self.embedding_dim, bias=False
        )
        self.embedding_positional = nn.Parameter(
            torch.randn((self.sequence_length, self.embedding_dim)), requires_grad=True
        )
        self.cls_token = nn.Parameter(
            torch.randn((1, self.embedding_dim)), requires_grad=True
        )

        # Readout
        self.readout = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

        # Module list for layers
        self.layers = nn.ModuleList([])
        for _ in range(self.num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        MultiHeadSelfAttention(self.embedding_dim, self.num_heads),
                        FeedForward(self.embedding_dim),
                    ]
                )
            )

    def _embedding(self, x):
        # PERF: Might need to add LayerNorm here if the initializations are bad
        # Input x is C x H x W image. We assume H=W and C=3.
        patchified = rearrange(
            x,
            "b c (p1 h) (p2 w) -> b (p1 p2) (c h w)",
            p1=self.input_dim // self.patch_size,
            p2=self.input_dim // self.patch_size,
        )  # patchified is b x n x d_in now
        embedding = (
            self.patch_embedding_layernorm(self.embedding(patchified))
            + self.embedding_positional
        )  # b x n x d
        # Concatenate CLS token.
        cls_repeated = self.cls_token.unsqueeze(0).expand(x.shape[0], -1, -1)
        embedding = torch.concatenate(
            (cls_repeated, embedding), dim=1
        )  # b x (n + 1) x d
        return embedding

    def _readout(self, x):
        cls_token = x[:, 0, :]  # b x d
        return self.readout(cls_token)

    def forward(self, x):
        x = self._embedding(x)
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x
        x = self._readout(x)
        return x


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    # config and logging
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="test_train_cifar10", name="vit-test", config=config_dict)
    device = cfg.training.device

    # data: transforms and loaders
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    )
    transform_lenet = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_val_lenet = transforms.Compose([transforms.ToTensor(), normalize])
    transform_vit = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_val_vit = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform = transform_vit
    transform_val = transform_val_vit

    train_data = CIFAR10(root="data/", train=True, transform=transform, download=True)
    val_data = CIFAR10(
        root="data/", train=False, transform=transform_val, download=True
    )
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=len(val_data),
        num_workers=8,
        pin_memory=True,
    )

    # model
    # CIFAR10 is 3 x 32 x 32 images.
    # There are 10 classes.

    # A linear model, to get things rolling
    # model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))

    # A simple convolutional model, like LeNet.
    # Follow https://github.com/soapisnotfat/pytorch-cifar10/blob/master/models/LeNet.py
    # model = nn.Sequential(
    #     nn.Conv2d(3, 6, kernel_size=5),  # 32 - 5 + 1 = 28
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2),  # 28 / 2 = 14
    #     nn.Conv2d(6, 16, kernel_size=5),  # 14-5+1 = 10
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2),  # 10 / 2 = 5. now at 16x5x5
    #     nn.Flatten(),
    #     nn.Linear(16 * 5 * 5, 120),
    #     nn.ReLU(),
    #     nn.Linear(120, 84),
    #     nn.ReLU(),
    #     nn.Linear(84, 10),
    # )

    # A simple vision transformer
    model = VisionTransformer(
        input_dim=224,
        embedding_dim=cfg.model.embedding_dim,
        patch_size=cfg.model.patch_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        num_classes=cfg.model.num_classes,
    )
    model = torch.compile(model)
    # TODO: Benchmarking for performance? torch.utils.benchmark

    # loss, optim, train loop
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr_initial,
        weight_decay=cfg.training.weight_decay,
    )
    # For doing LR scheduling:
    # 1. We create a scheduler object that wraps the optimizer.
    # 2. We optimize as usual, but we step the scheduler at certain key epochs (outer loop).
    #    The scheduler will be initialized with certain schedule parameters. Step takes the epoch as argument.
    #    (oops, this is depracated)
    # General intuition about scheduling:
    # A) start large. if initial training is unstable, consider (linear) warmup phase, going up to the initial large LR.
    # B) decay the LR over training with the scheduler.
    # C) for vision, consider using restarting on the LR decay

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=cfg.training.epochs
    # )
    # Customized cosine scheduler with warmup.
    # From the CRATE repo.
    lr_func = lambda epoch: min(
        (epoch + 1) / (cfg.training.warmup_epochs + 1e-8),
        0.5 * (math.cos(epoch / cfg.training.epochs * math.pi) + 1),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_func, verbose=True
    )

    model.to(device)
    for epoch in range(cfg.training.epochs):
        model.train()
        for batch, (X, Y) in enumerate(train_loader):
            # do work
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            loss, acc = process_batch(X, Y, loss_fn, model, optimizer, training=True)
            wandb.log(
                {"batch": batch, "batch_loss_train": loss, "batch_acc_train": acc}
            )
        # Step the scheduler
        scheduler.step()
        # Compute val loss on epoch
        model.eval()
        with torch.no_grad():
            for X, Y in val_loader:
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                loss, acc = process_batch(
                    X, Y, loss_fn, model, optimizer, training=False
                )
                wandb.log(
                    {"epoch": epoch, "epoch_loss_val": loss, "epoch_acc_val": acc}
                )

    # eval and logging
    wandb.finish()


def process_batch(
    X, Y, loss_fn, model, optimizer: torch.optim.Optimizer, training=True
):
    logits = model(X)
    preds = torch.argmax(logits, dim=-1)
    loss = loss_fn(logits, Y)
    acc = torch.sum(preds == Y) / Y.shape[0]

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss, acc


if __name__ == "__main__":
    train()
