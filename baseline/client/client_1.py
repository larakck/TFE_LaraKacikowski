import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from utils.model import UNet
from utils.dataset import get_dataloaders
from utils.train_eval import train, evaluate
import sys
import os
import argparse

try:
    from opacus import PrivacyEngine
except ImportError:
    PrivacyEngine = None  # pour éviter l'import si pas installé

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fonction de perte : Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        smooth = 1.0
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        return 1 - ((2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth))

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, noise_multiplier=0.0):
        self.model = UNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = DiceLoss()
        self.train_dl, self.val_dl = get_dataloaders("multicenter/external/Dataset004_SierraLeone", augment=False)

        self.privacy_engine = None
        if noise_multiplier > 0.0:
            print(f"[Client] Differential Privacy ACTIVÉE (nm = {noise_multiplier})")
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                noise_multiplier=noise_multiplier,
                max_grad_norm=1.0,
                poisson_sampling=False
            )
        else:
            print("[Client] Differential Privacy DÉSACTIVÉE (baseline)")

    def get_parameters(self, config=None):
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new, dtype=torch.float32).to(DEVICE)

    def fit(self, parameters, config):
        print("[Client] Start training...")
        self.set_parameters(parameters)
        loss = train(self.model, self.train_dl, self.optimizer, self.criterion)
        dice = evaluate(self.model, self.val_dl)
        epsilon = self.privacy_engine.get_epsilon(1e-5) if self.privacy_engine else None
        print(f"[Client] Training finished | Loss: {loss:.4f} | Dice: {dice:.4f}" +
              (f" | ε = {epsilon:.2f}" if epsilon else ""))
        return self.get_parameters(), len(self.train_dl.dataset), {
            "loss": float(loss),
            "dice": float(dice),
            "epsilon": float(epsilon) if epsilon else 0.0
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        dice = evaluate(self.model, self.val_dl)
        print(f"[Client] Evaluation finished | Dice: {dice:.4f}")
        return 1 - dice, len(self.val_dl.dataset), {"dice": float(dice)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_multiplier", type=float, default=0.0, help="Noise multiplier for DP")
    args = parser.parse_args()

    fl.client.start_client(
        server_address="localhost:8080",
        client=FlowerClient(noise_multiplier=args.noise_multiplier).to_client()
    )
