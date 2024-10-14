#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from einops import rearrange
from torch import nn
from torch.nn import Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

"""Simple convnet on mnist (pytorch tutorial)"""


class SimpleFFNN(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.flatten = Flatten()
        self.layers = Sequential(
            Linear(28 * 28, 1024), ReLU(), Linear(1024, 512), ReLU(), Linear(512, 10)
        )
        # initializations
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


def train(
    batch_size=32,
    num_epochs=1024,
    device=torch.device("cpu"),
    weight_decay=0.0,
    lr=1e-3,
):
    # config and logging

    # data
    # MNIST images are already in Tensor format.
    # Targets are in {0, 1, ..., 9}.
    dataset_train = MNIST("data/", train=True, download=True, transform=ToTensor())
    num_train = len(dataset_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dataset_val = MNIST("data/", train=False, download=False, transform=ToTensor())
    num_val = len(dataset_val)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=num_val,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # model
    model = SimpleFFNN().to(device)

    # training loop
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    running_loss = 0.
    for epoch in range(num_epochs):
        for batch, (X, Y) in enumerate(dataloader_train):
            X, Y = X.to(device), Y.to(device)
            loss = process_batch(X, Y, model, loss_fn, optimizer, train=True)
            running_loss += loss
        print(f"epoch {epoch}, loss {running_loss / (num_train / batch_size)}")
        running_loss = 0.

    # eval and logging
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader_val:
            X, Y = X.to(device), Y.to(device)
            loss = process_batch(X, Y, model, loss_fn, optimizer, train=False)
            test_preds = torch.argmax(model(X), dim=-1)
            correct_mask = (test_preds == Y)
            print(f"test loss {loss}, accuracy {correct_mask.sum() / num_val}")
    print('done')


def process_batch(X, Y, model, loss_fn, optimizer: torch.optim.Optimizer, train=True):
    preds = model(X)
    loss = loss_fn(preds, Y)

    if train:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss


if __name__ == "__main__":
    config = {
        "num_epochs": 8,
        "batch_size": 128,
        "device": torch.device("mps"),
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    train(**config)
