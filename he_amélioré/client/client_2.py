# client2.py

import flwr as fl
import torch
import numpy as np
import gc
import psutil
import math
from torch import nn, optim
from flwr.common import parameters_to_ndarrays

from utils.model import UNet
from utils.dataset import get_dataloaders
from utils.train_eval import train, evaluate, set_model_parameters
from he_amélioré.utils.encryption_utils import (
    create_ckks_context,
    encrypt_model_parameters,
    decrypt_model_parameters,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiceLoss(nn.Module):
    def forward(self, preds, targets):
        preds   = torch.sigmoid(preds).view(-1)
        targets = targets.view(-1)
        inter   = (preds * targets).sum()
        smooth  = 1.0
        return 1 - ((2 * inter + smooth) / (preds.sum() + targets.sum() + smooth))

class FlowerClient2(fl.client.NumPyClient):
    def __init__(self):
        # ——— Modèle & opti ———
        self.model     = UNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = DiceLoss()
        # ——— Dataset hard‑codé pour Client 2 ———
        self.train_dl, self.val_dl = get_dataloaders(
            "multicenter/train/Dataset002_Egypt", augment=False
        )
        # ——— Contexte CKKS ———
        self.context = create_ckks_context()
        # (on reconstruit shapes/chunk_counts à la volée dans set_parameters)

    def get_parameters(self, config=None):
        # Exporte les poids en clair, puis chiffre et sérialise
        weights = [val.cpu().numpy() for val in self.model.state_dict().values()]
        encrypted_chunks, _ = encrypt_model_parameters(weights, self.context)
        return [np.frombuffer(c, dtype=np.uint8) for c in encrypted_chunks]

    def set_parameters(self, parameters):
        print("[CLIENT 2] 🔓 Déchiffrement des paramètres reçus")
        # 1) Reconstruire shapes
        shapes = [p.shape for p in self.model.state_dict().values()]
        # 2) Calculer slot_count = degree//2
        slots = self.context.poly_modulus_degree // 2
        # 3) Déduire chunk_counts
        chunk_counts = [
            math.ceil(math.prod(shape) / slots)
            for shape in shapes
        ]
        # 4) Déchiffrement
        decrypted = decrypt_model_parameters(parameters, self.context, shapes, chunk_counts)
        print(f"[CLIENT 2 DEBUG] Poids[0]= {decrypted[0].flatten()[0]:.6f}")
        # 5) Moyenne en clair (n_clients = 2)
        averaged = [w / 2 for w in decrypted]
        set_model_parameters(self.model, averaged)
        print("[CLIENT 2] 🚀 Poids moyennés appliqués (n_clients=2)")

    def fit(self, parameters, config):
        # Applique les paramètres agrégés
        self.set_parameters(parameters)
        # Entraînement local
        print(f"[CLIENT 2 DEBUG] Avant entraînement, w00= {next(self.model.parameters()).view(-1)[0].item():.6f}")
        loss = train(self.model, self.train_dl, self.optimizer, self.criterion)
        dice = evaluate(self.model, self.val_dl)
        print(f"[CLIENT 2 DEBUG] Après entraînement, w00= {next(self.model.parameters()).view(-1)[0].item():.6f}")
        print(f"[CLIENT 2] 📊 Fit terminé | Loss={loss:.4f} | Dice={dice:.4f}")
        # Re‑chiffrement
        weights_np = [p.data.detach().cpu().numpy() for p in self.model.parameters()]
        encrypted_chunks, _ = encrypt_model_parameters(weights_np, self.context)
        encrypted_np = [np.frombuffer(c, dtype=np.uint8) for c in encrypted_chunks]
        gc.collect()
        print(f"[CLIENT 2] 🧠 RAM utilisée : {psutil.virtual_memory().used / (1024**3):.2f} GB")
        return encrypted_np, len(self.train_dl.dataset), {"loss": float(loss), "dice": float(dice)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        dice = evaluate(self.model, self.val_dl)
        print(f"[CLIENT 2] 📊 Eval | Dice={dice:.4f}")
        return 1 - dice, len(self.val_dl.dataset), {"dice": float(dice)}


if __name__ == "__main__":
    fl.client.start_client(
        server_address="localhost:8080",
        client=FlowerClient2().to_client(),
        grpc_max_message_length=2_147_483_647,
    )
