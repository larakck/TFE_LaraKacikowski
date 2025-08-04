import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import flwr as fl

from utils2.model import UNet
from utils2.train_eval import train_model_client
from utils2.metrics import dice_score, pixel_accuracy, iou_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Hyperparams ===
EPOCHS = 1
BATCH_SIZE = 16
LR = 1e-3
MAX_GRAD_NORM = 2.0

# === Dataset ===
BASE_DIR = "1327317/training_set_processed/clients_split/client1"
IMG_DIR = os.path.join(BASE_DIR, "imagesTr")
MASK_DIR = os.path.join(BASE_DIR, "labelsTr")

class FetalHCDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size

        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith('.png') and '_Annotation' not in f
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        mask_filename = img_filename.replace('.png', '_Annotation.png')

        img_path = os.path.join(self.images_dir, img_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.resize(image, self.target_size) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = cv2.resize(mask, self.target_size) / 255.0

        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        if self.transform:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)

            random.seed(seed)
            torch.manual_seed(seed)
            mask_transform = transforms.Compose([
                t for t in self.transform.transforms if not isinstance(t, transforms.Normalize)
            ])
            mask = mask_transform(mask)

        return image, mask

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ])

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.criterion = lambda p, t: 0.5 * self.bce(p, t) + 0.5 * self.dice_loss(p, t)

        self.train_dl, self.val_dl = get_dataloaders(
            "multicenter/external/Dataset004_SierraLeone",
            batch_size=batch_size,
            augment=True
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
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
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for val, param in zip(parameters, self.model.parameters()):
            param.data = torch.tensor(val, device=device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        noise_multiplier = config.get("noise_multiplier", 0.8)
        print(f"[CLIENT] Received noise_multiplier = {noise_multiplier}")

        train_loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        optimizer = optim.Adam(self.model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=8, cooldown=3, min_lr=1e-5
        )

        sample_rate = BATCH_SIZE / len(self.dataset)

        # Réutiliser le même PrivacyEngine pour accumuler
        if self.cumulative_privacy_engine is None:
            self.cumulative_privacy_engine = PrivacyEngine(accountant="rdp")
        self.model.train()
        self.model, optimizer, train_loader = self.cumulative_privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=MAX_GRAD_NORM,
            sample_rate=sample_rate
        )

        self.model, history = train_model_client(
            model=self.model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=self.criterion,
            epochs=EPOCHS,
            device=device
        )

        try:
            self.epsilon = self.cumulative_privacy_engine.accountant.get_epsilon(delta=self.delta)
        except Exception as e:
            print(f"[CLIENT] Failed to compute epsilon: {e}")
            self.epsilon = 0.0

        final_loss = history['loss'][-1] if history['loss'] else 1.0
        final_dice = history['dice'][-1] if history['dice'] else 0.0
        final_acc = history['accuracy'][-1] if history['accuracy'] else 0.0

        print(f"[CLIENT] Final metrics | Loss: {final_loss:.4f} | Dice: {final_dice:.4f} | Acc: {final_acc:.4f} | ε = {self.epsilon:.4f}")

        return self.get_parameters(), len(self.dataset), {
            "loss": final_loss,
            "dice": final_dice,
            "accuracy": final_acc,
            "epsilon": self.epsilon,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_dice = 0.0
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE)
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = self.model(x)
                total_dice += dice_score(pred, y).item()
        avg_dice = total_dice / len(loader)
        print(f"[CLIENT] Eval Dice Score: {avg_dice:.4f}")
        return 0.0, len(self.dataset), {"dice": avg_dice}

# === Lancer le client ===
if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FLClient())

