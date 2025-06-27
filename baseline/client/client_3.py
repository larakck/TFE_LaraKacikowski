
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from utils.model import UNet
from utils.dataset import get_dataloaders
from utils.train_eval import train, evaluate
import sys
import os

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
    def __init__(self):
        self.model = UNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = DiceLoss()
        # Activation de l'augmentation car dataset plus petit
        self.train_dl, self.val_dl = get_dataloaders("multicenter/train/Dataset001_Algeria", augment=True)

    def get_parameters(self, config=None):
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new, dtype=torch.float32)

    def fit(self, parameters, config):
        print("[Client] Start training...")
        self.set_parameters(parameters)
        loss = train(self.model, self.train_dl, self.optimizer, self.criterion)
        dice = evaluate(self.model, self.val_dl)
        print(f"[Client] Training finished | Loss: {loss:.4f} | Dice: {dice:.4f}")
        return self.get_parameters(), len(self.train_dl.dataset), {"loss": float(loss), "dice": float(dice)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        dice = evaluate(self.model, self.val_dl)
        print(f"[Client] Evaluation finished | Dice: {dice:.4f}")
        return 1 - dice, len(self.val_dl.dataset), {"dice": float(dice)}

if __name__ == "__main__":
    fl.client.start_client(server_address="localhost:8080", client=FlowerClient().to_client())
