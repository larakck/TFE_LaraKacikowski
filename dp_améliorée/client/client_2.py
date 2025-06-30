import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from utils.model import UNet
from utils.dataset import get_dataloaders
from utils.train_eval import train, evaluate
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dice Loss (soft version)
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        smooth = 1e-6
        preds = torch.sigmoid(preds).view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (preds * targets).sum(dim=1)
        sums = preds.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + smooth) / (sums + smooth)
        return 1 - dice.mean()


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        local_epochs=5,
        noise_multiplier=1.0,
        batch_size=8,
        max_grad_norm=0.8,
        lr=1e-3
    ):
        self.model = UNet().to(DEVICE)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.criterion = lambda p, t: 0.5 * self.bce(p, t) + 0.5 * self.dice_loss(p, t)

        self.train_dl, self.val_dl = get_dataloaders(
            "multicenter/train/Dataset002_Egypt",
            batch_size=batch_size,
            augment=True
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # DP attributes
        self.privacy_engine = None
        self.dp_enabled = False
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.sample_rate = batch_size / len(self.train_dl.dataset)
        self.max_grad_norm = max_grad_norm
        self.local_epochs = local_epochs

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new, dtype=torch.float32).to(DEVICE)

    def fit(self, parameters, config):
        # Récupération dynamique du noise_multiplier depuis le serveur
        nm = config.get("noise_multiplier", self.noise_multiplier)
        # Si le noise_multiplier a changé, réinitialiser la DP
        if nm != self.noise_multiplier:
            self.noise_multiplier = nm
            self.dp_enabled = False

        self.set_parameters(parameters)

        # Activer DP si besoin
        if self.noise_multiplier > 0 and not self.dp_enabled:
            self.model.train()  # Important pour Opacus
            # (Ré)initialisation du PrivacyEngine avec le nouveau noise_multiplier
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                sample_rate=self.sample_rate
            )
            self.dp_enabled = True

        loss = 0.0
        for _ in range(self.local_epochs):
            loss = train(self.model, self.train_dl, self.optimizer, self.criterion)
            self.scheduler.step(loss)

        dice = evaluate(self.model, self.val_dl)

        # Calcul correct d'ε via l'accountant
        epsilon = (
            self.privacy_engine.accountant.get_epsilon(delta=1e-5)
            if self.privacy_engine else 0.0
        )

        print(f"[Client] Final Results -> Loss: {loss:.4f} | Dice: {dice:.4f} | ε = {epsilon:.2f}")
        return self.get_parameters(), len(self.train_dl.dataset), {
            "loss": float(loss),
            "dice": float(dice),
            "epsilon": float(epsilon),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        dice = evaluate(self.model, self.val_dl)
        print(f"[Client] Evaluation Dice: {dice:.4f}")
        return 1 - dice, len(self.val_dl.dataset), {"dice": float(dice)}


if __name__ == "__main__":
    client = FlowerClient(
        local_epochs=2,
        noise_multiplier=1.0,   # Valeur par défaut
        batch_size=8,
        max_grad_norm=0.8,
        lr=1e-3,
    )
    try:
        fl.client.start_client(server_address="localhost:8080", client=client.to_client())
    except Exception as e:
        print(f"[Client] Connection error: {e}")
        sys.exit(1)
