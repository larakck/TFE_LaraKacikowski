import flwr as fl
import torch
import numpy as np
import os
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
        preds = torch.sigmoid(preds).view(-1)
        targets = targets.view(-1)
        inter = (preds * targets).sum()
        smooth = 1.0
        return 1 - ((2 * inter + smooth) / (preds.sum() + targets.sum() + smooth))

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        # ==== Inchangé ====
        self.model = UNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = DiceLoss()
        self.train_dl, self.val_dl = get_dataloaders(
            "multicenter/external/Dataset004_SierraLeone", augment=False
        )
        # =================
        # Création du contexte CKKS
        self.context = create_ckks_context()
        # Variables pour shape/chunk_counts
        self.shapes = None
        self.chunk_counts = None

    def get_parameters(self, config=None):
        # ==== Inchangé ====
        weights = [val.cpu().numpy() for val in self.model.state_dict().values()]
        self.shapes = [w.shape for w in weights]
        encrypted_chunks, self.chunk_counts = encrypt_model_parameters(
            weights, self.context
        )
        return [np.frombuffer(chunk, dtype=np.uint8) for chunk in encrypted_chunks]
        # =================

    def set_parameters(self, parameters):
        print("[CLIENT] 🔓 Déchiffrement des paramètres reçus")

        # Vérifie que shapes/chunk_counts ont été initialisés
        if self.shapes is None or self.chunk_counts is None:
            print("[CLIENT] ❌ shapes ou chunk_counts manquants, impossible de déchiffrer")
            raise ValueError("Missing shapes or chunk_counts for decryption")

        try:
            # Déchiffrement homomorphe
            decrypted = decrypt_model_parameters(
                parameters, self.context, self.shapes, self.chunk_counts
            )
            print(f"[DEBUG] 🔍 Poids reçus (0,0,0,0) : {decrypted[0].flatten()[0]}")

            # ==== MODIF : moyenne en clair ====
            n_clients = 1  # Remplacez par le nombre réel de clients si >1
            averaged = [w / n_clients for w in decrypted]
            set_model_parameters(self.model, averaged)
            print(f"[CLIENT] 🚀 Poids moyennés appliqués (n_clients={n_clients})")
            # ====================================
        except Exception as e:
            print(f"[CLIENT] ❌ Erreur de déchiffrement : {e}")
            raise e

    def fit(self, parameters, config):
        # Applique les paramètres agrégés
        self.set_parameters(parameters)

        # ==== Inchangé ====
        print(f"[DEBUG] 🔁 Avant entraînement, poids[0][0][0][0] : "
              f"{self.model.parameters().__next__().view(-1)[0].item()}")
        loss = train(self.model, self.train_dl, self.optimizer, self.criterion)
        dice = evaluate(self.model, self.val_dl)
        print(f"[DEBUG] ✅ Après entraînement, poids[0][0][0][0] : "
              f"{self.model.parameters().__next__().view(-1)[0].item()}")
        print(f"[Client] 📊 Fit terminé | Loss={loss:.4f} | Dice={dice:.4f}")

        # Rechiffrement des poids mis à jour
        weights_np = [p.data.detach().cpu().numpy() for p in self.model.parameters()]
        self.shapes = [w.shape for w in weights_np]
        encrypted_chunks, self.chunk_counts = encrypt_model_parameters(
            weights_np, self.context
        )
        encrypted_np = [np.frombuffer(c, dtype=np.uint8) for c in encrypted_chunks]

        del weights_np, encrypted_chunks
        gc.collect()
        print(f"[Client] 🧠 RAM utilisée : {psutil.virtual_memory().used / (1024**3):.2f} GB")

        return encrypted_np, len(self.train_dl.dataset), {"loss": float(loss), "dice": float(dice)}
        # =================

    def evaluate(self, parameters, config):
        # Applique et évalue  
        self.set_parameters(parameters)
        print("[DEBUG] 🧪 Début évaluation modèle")
        dice = evaluate(self.model, self.val_dl)
        print(f"[Client] 📊 Eval | Dice={dice:.4f}")
        return 1 - dice, len(self.val_dl.dataset), {"dice": float(dice)}

if __name__ == "__main__":
    fl.client.start_client(
        server_address="localhost:8080",
        client=FlowerClient().to_client(),
        grpc_max_message_length=2_147_483_647,
    )
