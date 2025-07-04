# he/client/client_3.py

import os, sys
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from he.utils.model import UNet
from he.utils.dataset import get_dataloaders
from he.utils.train_eval import train, evaluate
from he.utils.encryption_utils import (
    create_ckks_context,
    encrypt_model_parameters,
    decrypt_model_parameters,
)

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds).view(-1)
        targets = targets.view(-1)
        inter = (preds * targets).sum()
        smooth = 1.0
        return 1 - ((2 * inter + smooth) / (preds.sum() + targets.sum() + smooth))

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        # 1) Modèle & train/eval setup
        self.model = UNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = DiceLoss()
        self.train_dl, self.val_dl = get_dataloaders(
            "multicenter/train/Dataset001_Algeria", augment=True
        )

        # 2) Contexte CKKS
        self.private_context = create_ckks_context()
        self.public_context = self.private_context.copy()
        self.public_context.make_context_public()

        self.round = 0

    def fit(self, parameters, config):
        # 0) Si round > 0, lire & déchiffrer le fichier agrégé
        if self.round > 0:
            fname = f"weights/encrypted_agg_round{self.round}.bin"
            print(f"[Client] Round {self.round+1}: lecture {fname}...")
            # parser le fichier
            with open(fname, "rb") as f:
                data = f.read()
            chunks = []
            i = 0
            while i < len(data):
                length = int.from_bytes(data[i:i+4], "big"); i += 4
                chunks.append(data[i : i + length]); i += length
            # déchiffrement
            decrypted = decrypt_model_parameters(chunks, self.private_context)
            for p, arr in zip(self.model.parameters(), decrypted):
                p.data = torch.tensor(arr.reshape(p.shape), dtype=torch.float32)
            print(f"[Client] Round {self.round+1}: poids appliqués")

        # 1) Entraînement local
        print(f"[Client] Round {self.round+1}: start training...")
        loss = train(self.model, self.train_dl, self.optimizer, self.criterion)
        dice = evaluate(self.model, self.val_dl)
        print(f"[Client] Round {self.round+1}: Loss={loss:.4f} | Dice={dice:.4f}")

        # 2) Chiffrement des poids entraînés
        weights_np = [p.detach().cpu().numpy() for p in self.model.parameters()]
        encrypted = encrypt_model_parameters(weights_np, self.public_context)
        # écrire dans un fichier
        out = f"weights/encrypted_client_round{self.round+1}.bin"
        with open(out, "wb") as f:
            for chunk in encrypted:
                f.write(len(chunk).to_bytes(4, "big"))
                f.write(chunk)
        print(f"[Client] Round {self.round+1}: encrypted -> {out}")

        self.round += 1
        # 3) Retour light : pas de gros parameters, juste métriques
        return [], len(self.train_dl.dataset), {"loss": float(loss), "dice": float(dice), "file": out}

    def evaluate(self, parameters, config):
        # Pour l'éval, on applique la même logique que fit pour déchiffrement
        fname = f"weights/encrypted_agg_round{self.round}.bin"
        print(f"[Client] Eval Round {self.round}: lecture {fname}...")
        with open(fname, "rb") as f:
            data = f.read()
        chunks = []
        i = 0
        while i < len(data):
            length = int.from_bytes(data[i:i+4], "big"); i += 4
            chunks.append(data[i : i + length]); i += length
        decrypted = decrypt_model_parameters(chunks, self.private_context)
        for p, arr in zip(self.model.parameters(), decrypted):
            p.data = torch.tensor(arr.reshape(p.shape), dtype=torch.float32)

        dice = evaluate(self.model, self.val_dl)
        print(f"[Client] Eval Round {self.round}: Dice={dice:.4f}")
        return 1 - dice, len(self.val_dl.dataset), {"dice": float(dice)}

if __name__ == "__main__":
    fl.client.start_client(server_address="localhost:8080", client=FlowerClient().to_client())
