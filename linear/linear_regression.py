#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections.abc import Callable
from time import sleep

import torch
from torch.nn import Linear, MSELoss
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(
        self,
        data_dim: int,
        num_samples: int,
        ground_truth: torch.nn.Linear | None = None,
        device: torch.device = torch.device("cpu"),
        noise_std: float = 0.1,
        transform: Callable | None = None,
        # target_transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.data = torch.randn(
            (num_samples, data_dim), device=device, dtype=torch.float32
        )
        self.noise_std = noise_std
        if ground_truth is not None:
            self.ground_truth = ground_truth  # can't clone a pytorch module; need to deepcopy or manually
        else:
            self.ground_truth = torch.nn.Linear(
                data_dim, 1, device=device, dtype=torch.float32
            )
            torch.nn.init.normal_(self.ground_truth.weight)
            torch.nn.init.zeros_(self.ground_truth.bias)
            for parameter in self.ground_truth.parameters():
                parameter.requires_grad = False
        with torch.no_grad():
            self.targets = self.ground_truth(self.data) + self.noise_std * torch.randn(
                (num_samples, 1), device=device, dtype=torch.float32
            )
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx, :]
        if self.transform is not None:
            data = self.transform(data)
        return (data, self.targets[idx])


class LinearModel(torch.nn.Module):
    def __init__(
        self, data_dim: int, device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.linear = Linear(data_dim, 1, device=device, dtype=torch.float32)
        torch.nn.init.normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)


def train(
    data_dim,
    num_samples_train,
    num_samples_val,
    device,
    noise_std=0.1,
    batch_size=32,
    num_epochs=128,
    **kwargs,
):
    # TODO:
    # 1. logging (wandb)
    # We could do this with hydra. Chatgpt recommended a kwargs dict structure but hydra is better. Too much rewriting config table.
    # 2. get data
    train_dataset = RandomDataset(
        data_dim=data_dim,
        num_samples=num_samples_train,
        noise_std=noise_std,
        device=device,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=False, shuffle=True
    )
    val_dataset = RandomDataset(
        data_dim=data_dim,
        num_samples=num_samples_val,
        noise_std=noise_std,
        ground_truth=train_dataset.ground_truth,
    )
    val_dataloader = DataLoader(
        val_dataset, drop_last=False, shuffle=True, batch_size=num_samples_val
    )
    # 3. get model
    model = LinearModel(data_dim, device)
    # 4. train
    loss_fn = MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1, weight_decay=0.0)
    model.train()
    for epoch in range(num_epochs):
        for batch, (X, Y) in enumerate(
            train_dataloader
        ):  # enumerate(iterable) gives us the batch idx.
            loss = process_batch(model, X, Y, loss_fn, optimizer)
            print(f"batch {batch}, epoch {epoch}, loss: {loss.item()}")
    # 5. Tests and logging.
    model.eval()
    print(
        f"parameter error: {torch.sum((model.linear.weight - train_dataset.ground_truth.weight)**2)}"
    )
    with torch.no_grad():
        for X, Y in val_dataloader:
            loss = process_batch(model, X, Y, loss_fn, optimizer, training=False)
            print(f"validation loss: {loss.detach()}")
    sleep(1)


def process_batch(model, X, Y, loss_fn, optimizer, training=True):
    preds = model(X)
    loss = loss_fn(preds, Y)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.detach()


if __name__ == "__main__":
    num_samples_train = 128
    config = {
        "data_dim": 16,
        "num_samples_train": num_samples_train,
        "num_samples_val": 128,
        "noise_std": 0.0,
        "batch_size": num_samples_train,
        "num_epochs": 2**10,
        "device": torch.device("cpu"),
    }
    train(**config)
