# client/client_1.py
import flwr as fl
import torch, torch.nn as nn, torch.optim as optim
from utils.model import UNet
from utils.dataset import get_dataloaders
from utils.train_eval import train, evaluate
from he.utils.encryption_utils import create_ckks_context, encrypt_model_parameters, decrypt_model_parameters
import argparse, os, sys

try:
    from opacus import PrivacyEngine
except ImportError:
    PrivacyEngine = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiceLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds).view(-1)
        targets = targets.view(-1)
        inter = (preds * targets).sum()
        smooth = 1.0
        return 1 - ((2 * inter + smooth) / (preds.sum() + targets.sum() + smooth))

class HEFlowerClient(fl.client.NumPyClient):
    def __init__(self, noise=0.0):
        # initialisation normale
        self.model = UNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = DiceLoss()
        self.train_dl, self.val_dl = get_dataloaders(
            "multicenter/train/Dataset002_Egypt",
            augment=True
        )
        # DP si demandé
        self.privacy_engine = None
        if noise > 0.0 and PrivacyEngine:
            print(f"[Client] DP ACTIVÉE (nm={noise})")
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                noise_multiplier=noise,
                max_grad_norm=1.0,
                poisson_sampling=False
            )
        else:
            print("[Client] DP DÉSACTIVÉE")

        # Contexte HE partagé client ↔ serveur
        self.he_context = create_ckks_context()

    def get_parameters(self, config=None):
        # on chiffre et on renvoie des bytes
        return encrypt_model_parameters(self.model, self.he_context)

    def set_parameters(self, parameters):
        # on attend ici des poids clairs, déchiffrés par le serveur
        for p, new in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new, dtype=torch.float32).to(DEVICE)

    def fit(self, parameters, config):
        # paramètres reçus = poids du modèle global en clair
        self.set_parameters(parameters)
        loss = train(self.model, self.train_dl, self.optimizer, self.criterion)
        dice = evaluate(self.model, self.val_dl)
        eps = self.privacy_engine.get_epsilon(1e-5) if self.privacy_engine else None
        print(f"[Client] Fit | Loss={loss:.4f} | Dice={dice:.4f}"
              + (f" | ε={eps:.2f}" if eps else ""))
        # renvoie des poids chiffrés pour serveur
        return self.get_parameters(), len(self.train_dl.dataset), {
            "loss": float(loss),
            "dice": float(dice),
            "epsilon": float(eps) if eps else 0.0,
        }

    def evaluate(self, parameters, config):
        # pas de chiffrement pour l'éval, on reçoit déjà les poids clairs
        self.set_parameters(parameters)
        dice = evaluate(self.model, self.val_dl)
        print(f"[Client] Eval | Dice={dice:.4f}")
        return 1 - dice, len(self.val_dl.dataset), {"dice": float(dice)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.0, help="Noise multiplier DP")
    args = parser.parse_args()

    client = HEFlowerClient(noise=args.noise).to_client()
    fl.client.start_client(server_address="localhost:8080", client=client)
