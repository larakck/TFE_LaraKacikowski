# === utils/train_eval.py (modifié avec protections) ===

import torch
import torch.nn as nn
import torch.optim as optim
from .metrics import dice_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)

        # Vérifie si la loss est valide
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ERROR] Batch {batch_idx} - Invalid loss: {loss.item()}")
            continue

        try:
            loss.backward()
        except RuntimeError as e:
            print(f"[ERROR] RuntimeError during backward pass at batch {batch_idx}: {str(e)}")
            continue

        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"[TRAIN] Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

    print(f"[TRAIN] Epoch finished. Avg loss: {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader)


def evaluate(model, dataloader):
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = torch.sigmoid(model(x))
            total_dice += dice_score(pred, y)
    return total_dice / len(dataloader)

def set_model_parameters(model: torch.nn.Module, parameters: list[torch.Tensor]) -> None:
    for param, new_param in zip(model.parameters(), parameters):
        param.data = new_param.data.clone()
