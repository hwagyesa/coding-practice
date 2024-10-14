#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import wandb


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    # config and logging
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="test_train_cifar10", name="resnet18", config=config_dict)
    device = cfg.training.device

    # data
    train_data = CIFAR10(root="data/", train=True, transform=ToTensor(), download=True)
    val_data = CIFAR10(root="data/", train=False, transform=ToTensor(), download=True)
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
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))

    # loss, optim, train loop
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    for epoch in range(cfg.training.epochs):
        for X, Y in train_loader:
            # do work
            loss, acc = process_batch(X, Y, loss_fn, model, optimizer, training=True)
            wandb.log({"batch_loss_train": loss, "batch_acc_train": acc})
        # Compute val loss on epoch
        for X, Y in val_loader:
            loss, acc = process_batch(X, Y, loss_fn, model, optimizer, training=False)
            wandb.log({"epoch_loss_val": loss, "epoch_acc_val": acc})

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
