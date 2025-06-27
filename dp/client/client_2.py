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

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, targets):
        smooth = 1.0
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        inter = (preds * targets).sum()
        return 1 - ((2 * inter + smooth) / (preds.sum() + targets.sum() + smooth))

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = UNet().to(DEVICE)
        self.criterion = DiceLoss()
        # même code que client_1.py, sauf cette ligne :
        self.train_dl, self.val_dl = get_dataloaders("multicenter/train/Dataset002_Egypt", augment=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.privacy_engine = None
        self.dp_enabled = False

    def get_parameters(self, config=None):
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new, dtype=torch.float32).to(DEVICE)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        noise_multiplier = float(config.get("noise_multiplier", 0.0))
        print(f"[Client 1] noise_multiplier = {noise_multiplier}")
        if noise_multiplier > 0.0 and not self.dp_enabled:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                noise_multiplier=noise_multiplier,
                max_grad_norm=1.0
            )
            self.dp_enabled = True
        loss = train(self.model, self.train_dl, self.optimizer, self.criterion)
        dice = evaluate(self.model, self.val_dl)
        epsilon = self.privacy_engine.get_epsilon(1e-5) if self.privacy_engine else 0.0
        print(f"[Client 1] Loss: {loss:.4f} | Dice: {dice:.4f} | ε = {epsilon:.2f}")
        return self.get_parameters(), len(self.train_dl.dataset), {
            "loss": float(loss),
            "dice": float(dice),
            "epsilon": float(epsilon),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        dice = evaluate(self.model, self.val_dl)
        print(f"[Client 2] Evaluation Dice: {dice:.4f}")
        return 1 - dice, len(self.val_dl.dataset), {"dice": float(dice)}

if __name__ == "__main__":
    fl.client.start_client(server_address="localhost:8080", client=FlowerClient().to_client())
