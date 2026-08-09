import numpy as np
import torch
import torch.nn as nn


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        total_loss += criterion(model(X_batch), y_batch).item() * len(y_batch)
    return total_loss / len(loader.dataset)


def fit(model, train_loader, val_loader, epochs, lr, device):
    """Ciclo forward -> loss -> backward -> step por época, com validação e
    checkpoint do melhor state_dict (menor loss de validação)."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_loss = evaluate_loss(model, train_loader, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return history, best_state


@torch.no_grad()
def predict_probs(model, loader, device):
    """Roda o modelo em modo avaliação e converte logits (BCEWithLogitsLoss) em probabilidades."""
    model.eval()
    probs, targets = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        targets.extend(y_batch.numpy().flatten())
    return np.array(probs), np.array(targets)
