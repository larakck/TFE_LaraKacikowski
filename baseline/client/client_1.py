import os
import sys
import argparse
import torch
import flwr as fl
import torch.optim as optim

from utils2.model import UNet
from utils2.dataset import FetalHCDataset, get_train_transforms
from utils2.metrics import BCEDiceLoss, visualize_predictions
from utils2.train_eval import train_model_client, evaluate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, data_dir):
        self.model = UNet().to(DEVICE)
        self.criterion = BCEDiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        # Dataset
        images_dir = os.path.join(data_dir, "imagesTr")
        masks_dir = os.path.join(data_dir, "labelsTr")

        full_dataset = FetalHCDataset(
            images_dir,
            masks_dir,
            transform=get_train_transforms(),
            target_size=(256, 256)
        )

        # Split 80/20
        val_split = int(0.2 * len(full_dataset))
        train_split = len(full_dataset) - val_split
        self.train_dl, self.val_dl = torch.utils.data.random_split(
            full_dataset, [train_split, val_split], generator=torch.Generator().manual_seed(42)
        )

        self.train_dl = torch.utils.data.DataLoader(self.train_dl, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
        self.val_dl = torch.utils.data.DataLoader(self.val_dl, batch_size=4, shuffle=False, num_workers=0)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=5,
            min_lr=5e-6,
            cooldown=2
        )

    def get_parameters(self, config=None):
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new, dtype=torch.float32).to(DEVICE)


    def fit(self, parameters, config):
        print("[Client] Start training...")
        self.set_parameters(parameters)

        # Train for 1 epoch using the centralized train_model logic
        _, history = train_model_client(
            self.model,
            self.train_dl,
            self.optimizer,
            self.scheduler,
            self.criterion,
            epochs=1,
            device=DEVICE
        )

        val_loss, val_dice, val_acc, _ = evaluate(self.model, self.val_dl, self.criterion, DEVICE)

        return self.get_parameters(), len(self.train_dl.dataset), {
            "loss": float(history['loss'][-1]),
            "dice": float(val_dice),
            "accuracy": float(history['accuracy'][-1])
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, val_dice, val_acc, _ = evaluate(self.model, self.val_dl, self.criterion, DEVICE)

        # visualize_predictions(
        #     model=self.model,
        #     dataset=self.val_dl.dataset,
        #     device=DEVICE,
        #     num_samples=4,
        #     save_path="predictions_client_eval.png"
        # )

        print(f"[Client] Evaluation finished | Dice: {val_dice:.4f} | Acc: {val_acc:.4f}")
        return float(val_loss), len(self.val_dl.dataset), {
            "dice": float(val_dice),
            "accuracy": float(val_acc)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="1327317/training_set_processed", help="Path to dataset")
    args = parser.parse_args()

    fl.client.start_client(
        server_address="localhost:8080",
        client=FlowerClient(data_dir=args.data_dir).to_client()
    )
