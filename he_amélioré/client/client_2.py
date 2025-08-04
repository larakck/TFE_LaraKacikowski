# === client.py ===
import os
import torch
import flwr as fl
import tenseal as ts
import numpy as np
from utils.model import UNet
from utils.dataset import ClientDataset
from torch.utils.data import DataLoader

# Configuration
BASE_DATA_DIR = "1327317/training_set_processed"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
MODEL_PATH = "client_he_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HE Setup
def get_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 21, 21, 40],
    )
    context.global_scale = 2**21
    context.generate_galois_keys()
    return context

# Encrypt model weights
def encrypt_weights(model, context):
    encrypted = []
    for param in model.parameters():
        vec = param.detach().cpu().numpy().flatten().tolist()
        enc = ts.ckks_vector(context, vec)
        encrypted.append(enc.serialize())
    return encrypted

# Decrypt weights (for completeness, unused by client)
def decrypt_weights(encrypted, context, model):
    pointer = 0
    with torch.no_grad():
        for param in model.parameters():
            enc_tensor = ts.ckks_vector_from(context, encrypted[pointer])
            vec = torch.tensor(enc_tensor.decrypt(), dtype=torch.float32).view(param.shape)
            param.copy_(vec)
            pointer += 1

# Dice score
def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()

# Pixel accuracy
def pixel_accuracy(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    correct = (pred == target).sum().float()
    total = target.numel()
    return (correct / total).item() if total > 0 else 0.0

# Dummy training function
def train(model, train_loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_dice = 0.0
        running_acc = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, masks)
            loss.backward()
            optimizer.step()

            dice = dice_score(output, masks)
            acc = pixel_accuracy(output, masks)

            running_loss += loss.item()
            running_dice += dice
            running_acc += acc

            print(f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} | Dice: {dice:.4f} | Acc: {acc:.4f}")

# Flower Client
class HEClient(fl.client.NumPyClient):
    def __init__(self, context):
        self.context = context
        self.model = UNet()
        self.model.to(DEVICE)

    def get_parameters(self, config):
        dummy = []
        for param in self.model.parameters():
            zero_tensor = torch.zeros_like(param).detach().cpu().numpy().flatten().tolist()
            encrypted = ts.ckks_vector(self.context, zero_tensor)
            dummy.append(encrypted.serialize())
        return dummy

    def set_parameters(self, parameters):
        decrypt_weights(parameters, self.context, self.model)

    def fit(self, parameters, config):
        print("[CLIENT] Déchiffrement des poids globaux")
        self.set_parameters(parameters)

        print("[CLIENT] Entraînement local...")
        train_loader = DataLoader(ClientDataset(BASE_DATA_DIR, "train", 256, augment=True), batch_size=BATCH_SIZE)
        train(self.model, train_loader)

        print("[CLIENT] Chiffrement des nouveaux poids")
        encrypted_weights = encrypt_weights(self.model, self.context)
        return encrypted_weights, len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

if __name__ == "__main__":
    context = get_context()
    fl.client.start_numpy_client(server_address="localhost:8080", client=HEClient(context))