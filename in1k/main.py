#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import wandb


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    # config and logging
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="test_train_cifar10", name="lenet", config=config_dict)
    device = cfg.training.device

    # data
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    )
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_val = transforms.Compose([transforms.ToTensor(), normalize])

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
    # IN-1k is 3 x 224 x 224 images.
    # There are 1000 classes.

    # Implement ViT
    model = ViT()


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
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
