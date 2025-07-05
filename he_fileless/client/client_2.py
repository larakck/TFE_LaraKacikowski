import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc

from he.utils.model import UNet
from he.utils.dataset import get_dataloaders
from he.utils.train_eval import train, evaluate
from he.utils.encryption_utils import (
    create_ckks_context_shared,
    encrypt_model_parameters,
    decrypt_model_parameters,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiceLoss(nn.Module):
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds).view(-1)
        targets = targets.view(-1)
        inter = (preds * targets).sum()
        smooth = 1.0
        return 1 - ((2 * inter + smooth) / (preds.sum() + targets.sum() + smooth))

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = UNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = DiceLoss()
        self.train_dl, self.val_dl = get_dataloaders(
            "multicenter/train/Dataset002_Egypt", augment=True
        )
        self.ckks_context = create_ckks_context_shared()
        self.round = 0
        self.shapes = [p.shape for p in self.model.parameters()]
        print("[Client] CKKS context ready")

    def get_parameters(self, config):
        encrypted = encrypt_model_parameters(
            [p.detach().cpu().numpy() for p in self.model.parameters()],
            self.ckks_context,
            chunk_size=16384
        )
        print(f"[Client] → Encrypted {len(encrypted)} chunks")
        return encrypted

    def set_parameters(self, encrypted_layers):
        flat = decrypt_model_parameters(encrypted_layers, self.ckks_context)
        idx = 0
        for p, shape in zip(self.model.parameters(), self.shapes):
            n_params = torch.prod(torch.tensor(shape)).item()
            vals = flat[idx : idx + n_params]
            p.data = torch.tensor(vals, dtype=torch.float32).reshape(shape)
            idx += n_params
        gc.collect()

    def fit(self, parameters, config):
        if self.round > 0:
            self.set_parameters(parameters)

        loss = train(self.model, self.train_dl, self.optimizer, self.criterion)
        dice = evaluate(self.model, self.val_dl)
        self.round += 1
        gc.collect()
        return self.get_parameters(config), len(self.train_dl.dataset), {
            "loss": float(loss),
            "dice": float(dice),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        dice = evaluate(self.model, self.val_dl)
        return 1 - dice, len(self.val_dl.dataset), {"dice": float(dice)}

if __name__ == "__main__":
    fl.client.start_client(
        server_address="localhost:8080",
        client=FlowerClient().to_client(),
    )
