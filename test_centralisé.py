# test_plain_training.py
from utils.model import UNet
from utils.train_eval import train, evaluate
from utils.dataset import get_dataloaders
import torch
from torch import nn, optim

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

def main():
    net = UNet().to("cpu")
    train_dl, val_dl = get_dataloaders("multicenter/external/Dataset004_SierraLeone",
                                       augment=True, batch_size=8)
    optim  = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit   = DiceLoss()

    # Une ou deux époques uniquement
    for epoch in range(2):
        loss = train(net, train_dl, optim, crit)
        dice = evaluate(net, val_dl)
        print(f"Epoch {epoch} → Loss={loss:.4f} | Dice={dice:.4f}")

    if dice < 0.5:
        print("⚠️ Votre pipeline de training plain peine à dépasser Dice=0.5")
        exit(1)

    print("✔️ Training plain OK (Dice≃", dice, ")")

if __name__ == "__main__":
    main()
