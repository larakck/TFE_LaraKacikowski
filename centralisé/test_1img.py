import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

from utils.model import UNet
from utils.dataset import get_dataloaders  # ou directement ClientDataset si tu veux l’utiliser
from utils.metrics import dice_score

# === Configuration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "multicenter/external/Dataset004_SierraLeone"

# === Chargement du dataset ===
train_loader, _ = get_dataloaders(DATA_PATH, batch_size=4, img_size=128, augment=False)

# === DataLoader avec une seule image ===
one_sample_loader = DataLoader(Subset(train_loader.dataset, [0]), batch_size=1, shuffle=False)

# === Modèle ===
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Dice Loss seule ===
def dice_loss_only(probs, target, smooth=1e-6):
    intersection = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2. * intersection + smooth) / (union + smooth)).mean()

# === Entraînement sur 1 image ===
dice_history = []

for epoch in range(200):
    model.train()
    for x, y in one_sample_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        probs = torch.sigmoid(out)
        loss = dice_loss_only(probs, y)
        loss.backward()
        optimizer.step()

    # Évaluation (sur la même image)
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(x))
        print(f"pred shape: {pred.shape}, y shape: {y.shape}")

        dice = dice_score((pred > 0.5).float(), y)

        dice_history.append(dice.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} - Dice: {dice.item():.4f}")

# === Affichage de la courbe de convergence ===
plt.plot(dice_history)
plt.title("Dice sur une seule image")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.grid()
plt.show()

import matplotlib.pyplot as plt

x_np = x.cpu().squeeze().numpy()
y_np = y.cpu().squeeze().numpy()
pred_np = pred.cpu().squeeze().numpy()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(x_np, cmap="gray")
plt.title("Image")
plt.subplot(1, 3, 2)
plt.imshow(y_np, cmap="gray")
plt.title("Masque Ground Truth")
plt.subplot(1, 3, 3)
plt.imshow(pred_np, cmap="gray")
plt.title("Prédiction (Sigmoid)")
plt.tight_layout()
plt.show()
plt.close()
