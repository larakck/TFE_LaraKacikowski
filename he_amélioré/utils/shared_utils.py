# === utils/shared_utils.py ===

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from typing import Dict, List
import tenseal as ts
from utils.model import UNet  

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === 2. Fonctions de perte ===
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice_coeff

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice_loss(pred, target)


# === 3. Métriques ===
def dice_score(pred, target, smooth=1e-6):
    pred_sigmoid = torch.sigmoid(pred) if pred.requires_grad else pred
    pred_binary = (pred_sigmoid > 0.5).float()
    target_binary = (target > 0.5).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    if union == 0:
        return torch.tensor(1.0, device=pred.device)
    return (2. * intersection + smooth) / (union + smooth)


def pixel_accuracy(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    return (pred == target).sum() / target.numel()


# === 4. Dataset ===
class MulticenterFetalHCDataset(Dataset):
    def __init__(self, image_mask_pairs, transform=None, target_size=(256, 256)):
        self.image_mask_pairs = image_mask_pairs
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            image = np.zeros(self.target_size, dtype=np.uint8)
            mask = np.zeros(self.target_size, dtype=np.uint8)
        else:
            image = cv2.resize(image, self.target_size)
            mask = cv2.resize(mask, self.target_size)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        image = torch.from_numpy(image / 255.0).float().unsqueeze(0)
        mask = torch.from_numpy(mask / 255.0).float().unsqueeze(0)

        if self.transform:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            mask_transform = transforms.Compose(
                [t for t in self.transform.transforms if not isinstance(t, transforms.Normalize)]
            )
            mask = mask_transform(mask)

        return image, mask


def get_train_transforms():
    return transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ])


# === 5. TenSEAL HE Context ===
class TenSEALContext:
    def __init__(self, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]):
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, coeff_mod_bit_sizes)
        self.context.generate_galois_keys()
        self.context.global_scale = 2 ** 40


class ModelEncryptor:
    def __init__(self, context: TenSEALContext, encrypt_layers: List[str] = None):
        self.context = context.context
        self.encrypt_layers = encrypt_layers or []

    def should_encrypt_layer(self, name: str) -> bool:
        return not self.encrypt_layers or any(layer in name for layer in self.encrypt_layers)

    def encrypt_weights(self, state_dict: Dict) -> Dict:
        encrypted = {}
        for name, param in state_dict.items():
            flat = param.cpu().numpy().flatten()
            if self.should_encrypt_layer(name):
                if len(flat) <= 8192:
                    encrypted[name] = {'encrypted': ts.ckks_vector(self.context, flat), 'shape': param.shape, 'dtype': param.dtype}
                else:
                    chunks = [ts.ckks_vector(self.context, flat[i:i + 8192]) for i in range(0, len(flat), 8192)]
                    encrypted[name] = {'encrypted_chunks': chunks, 'chunk_size': 8192, 'total_size': len(flat),
                                       'shape': param.shape, 'dtype': param.dtype}
            else:
                encrypted[name] = {'plaintext': param.cpu(), 'shape': param.shape, 'dtype': param.dtype}
        return encrypted

    def decrypt_weights(self, encrypted_dict: Dict) -> Dict:
        decrypted = {}
        for name, data in encrypted_dict.items():
            if 'encrypted' in data:
                tensor = torch.tensor(data['encrypted'].decrypt(), dtype=data['dtype']).reshape(data['shape'])
                decrypted[name] = tensor
            elif 'encrypted_chunks' in data:
                flat = []
                for chunk in data['encrypted_chunks']:
                    flat.extend(chunk.decrypt())
                flat = flat[:data['total_size']]
                tensor = torch.tensor(flat, dtype=data['dtype']).reshape(data['shape'])
                decrypted[name] = tensor
            else:
                decrypted[name] = data['plaintext']
        return decrypted
