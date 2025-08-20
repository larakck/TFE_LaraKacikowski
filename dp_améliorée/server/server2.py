import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import gc
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from utils2.model import UNet
from utils2.train_eval import train_model_client
from utils2.metrics import dice_score, pixel_accuracy, iou_score

import time
import warnings
warnings.filterwarnings("ignore", message="PrivacyEngine detected new dataset object", category=UserWarning)
warnings.filterwarnings("ignore", message="Ignoring drop_last as it is not compatible with DPDataLoader.", category=UserWarning)

# === Config ===
CLIENT_DIRS = [
    "1327317/training_set_processed/clients_split/client1",
    "1327317/training_set_processed/clients_split/client2",
    "1327317/training_set_processed/clients_split/client3",
]

NUM_ROUNDS = 30
EPOCHS = 1
EPOCHS_CLIENT1 = 1  # ⬅️ client1 s’entraîne plus longtemps
BATCH_SIZE = 16
LR = 5e-4
MAX_GRAD_NORM = 2.0
NOISE_MULTIPLIER = 2.8
TARGET_SIZE = (256, 256)
VAL_RATIO  = 0.15
TEST_RATIO = 0.15 

# Eval tuning flags
ENABLE_TTA = True
ENABLE_POSTPROC = True
USE_TUNED_FOR_EARLYSTOP = True  # early stop basé sur Dice val "tuned" global

# Early stop config (sur GLOBAL on VAL)
best_global_val_dice = -1.0
best_global_state = None
best_round = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Logs (moyennes globales par round) ===
loss_per_round = []
dice_per_round = []
epsilon_per_round = []
val_loss_per_round = []
val_dice_per_round = []

# === Logs par client et par round ===
client_ids = [os.path.basename(p.rstrip("/")) for p in CLIENT_DIRS]
client_train_dice = {cid: [] for cid in client_ids}
client_train_loss = {cid: [] for cid in client_ids}
client_val_dice   = {cid: [] for cid in client_ids}
client_val_loss   = {cid: [] for cid in client_ids}

# Pour mémoriser les splits par client (val/test & éval global)
client_splits = {}

class FetalHCDataset(Dataset):
    """
    Dataset avec support d'indices (pour split).
    Augmentations synchronisées image/masque (sans Normalize sur le masque).
    """
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256), indices=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size

        all_files = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith('.png') and '_Annotation' not in f
        ])

        if indices is not None:
            self.image_files = [all_files[i] for i in indices]
        else:
            self.image_files = all_files

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
            # Synchronise image/mask
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed); torch.manual_seed(seed)
            image = self.transform(image)

            random.seed(seed); torch.manual_seed(seed)
            mask_transform = transforms.Compose([
                t for t in self.transform.transforms
                if not isinstance(t, transforms.Normalize)
            ])
            mask = mask_transform(mask)

        return image, mask

# import os
# import re
# import cv2
# import random
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms

# class FetalHCDataset(Dataset):
#     """
#     Dataset avec support d'indices (pour split).
#     Gère les masques:
#       1) <image>.png -> <image>_mask.png
#       2) 004_0000.png -> filled_004_gt_004.png (clé = 3 premiers chiffres)
#       3) fallback: <image>_Annotation.png

#     Augmentations synchronisées image/masque (sans Normalize sur le masque).

#     NB: Filtre silencieusement les items au masque vide pour éviter division par zéro
#         dans les métriques existantes, sans changer le reste du code.
#     """
#     def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256), indices=None):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.transform = transform
#         self.target_size = target_size

#         # --- Liste d'images (on ignore *_Annotation et *_mask) ---
#         all_files = sorted([
#             f for f in os.listdir(images_dir)
#             if f.endswith('.png') and '_Annotation' not in f and not f.endswith('_mask.png')
#         ])

#         if indices is not None:
#             self.image_files = [all_files[i] for i in indices]
#         else:
#             self.image_files = all_files

#         # --- Pré-indexation des masques ---
#         self.mask_by_base = {}  # base -> fichier masque (pour *_mask.png / *_Annotation.png)
#         self.mask_by_id3  = {}  # "004" -> fichier masque (pour filled_004_gt_004.png)

#         for f in os.listdir(masks_dir):
#             if not f.endswith('.png'):
#                 continue

#             if f.endswith('_mask.png'):
#                 base = f[:-len('_mask.png')]
#                 self.mask_by_base[base] = f
#                 continue

#             m = re.match(r'^filled_(\d{3})_gt_\1\.png$', f)
#             if m:
#                 id3 = m.group(1)
#                 self.mask_by_id3[id3] = f
#                 continue

#             if f.endswith('_Annotation.png'):
#                 base_anno = f[:-len('_Annotation.png')]
#                 # Ne pas écraser un *_mask existant si déjà présent
#                 self.mask_by_base.setdefault(base_anno, f)

#         self._leading_id3_regex = re.compile(r'^(\d{3})')

#         original = list(self.image_files)
#         kept = []
#         for fname in self.image_files:
#             mfile = self._find_mask_file(fname)
#             if mfile is None:
#                 continue
#             mpath = os.path.join(self.masks_dir, mfile)
#             mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
#             if mask is None:
#                 continue
#             mask_resized = cv2.resize(mask, self.target_size).astype(np.float32) / 255.0
#             # garde si au moins 1 pixel > 0
#             if (mask_resized > 0).any():
#                 kept.append(fname)

#         if len(kept) == 0:
#             # 🔁 Fallback: garder au moins tous les items qui ONT un fichier masque correspondant,
#             # même si ce masque est vide → évite len(dataset)==0
#             fallback = []
#             for fname in original:
#                 mfile = self._find_mask_file(fname)
#                 if mfile is not None and os.path.exists(os.path.join(self.masks_dir, mfile)):
#                     fallback.append(fname)
#             # s'il n'y a vraiment aucun masque correspondant, on laisse tel quel (DataLoader se plaindra logiquement)
#             self.image_files = fallback if len(fallback) > 0 else original
#         else:
#             self.image_files = kept

#     def __len__(self):
#         return len(self.image_files)

#     def _find_mask_file(self, img_filename: str):
#         """
#         Essaie, dans l'ordre :
#         1) <base>_mask.png (indexé) ou fichier réel s'il existe
#         2) <base_sans_suffixe4>_mask.png (ex: retire _0000 à la fin)
#         3) filled_<id3>_gt_<id3>.png où id3 est n'importe quel groupe de 3 chiffres dans le nom
#         4) <base>_Annotation.png ou <base_sans_suffixe4>_Annotation.png
#         """
#         base = img_filename[:-4]  # sans ".png"

#         # Variante sans suffixe slice "_0000" final éventuel
#         import re, os
#         base_wo_slice = re.sub(r'_\d{4}$', '', base)

#         # 1) *_mask via index
#         if base in self.mask_by_base:
#             return self.mask_by_base[base]
#         if base_wo_slice in self.mask_by_base:
#             return self.mask_by_base[base_wo_slice]

#         # 1bis) *_mask en test direct si pas indexé (robuste)
#         cand = base + "_mask.png"
#         if os.path.exists(os.path.join(self.masks_dir, cand)):
#             return cand
#         cand = base_wo_slice + "_mask.png"
#         if os.path.exists(os.path.join(self.masks_dir, cand)):
#             return cand

#         # 2) filled_###_gt_### via n'importe quel groupe de 3 chiffres dans le nom
#         m_any = re.search(r'(\d{3})', base)
#         if m_any:
#             id3 = m_any.group(1)
#             # d'abord via index si disponible
#             if id3 in self.mask_by_id3:
#                 return self.mask_by_id3[id3]
#             # sinon test direct d'existence
#             cand = f"filled_{id3}_gt_{id3}.png"
#             if os.path.exists(os.path.join(self.masks_dir, cand)):
#                 return cand

#         # 3) *_Annotation (fallback), version avec et sans suffixe slice
#         cand = base + "_Annotation.png"
#         if os.path.exists(os.path.join(self.masks_dir, cand)):
#             return cand
#         cand = base_wo_slice + "_Annotation.png"
#         if os.path.exists(os.path.join(self.masks_dir, cand)):
#             return cand

#         return None


#     def __getitem__(self, idx):
#         img_filename = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_filename)

#         mask_filename = self._find_mask_file(img_filename)
#         if mask_filename is None:
#             raise FileNotFoundError(
#                 f"Aucun masque correspondant trouvé pour {img_filename}. "
#                 f"Attendu *_mask.png ou filled_###_gt_###.png (ou *_Annotation.png)."
#             )
#         mask_path = os.path.join(self.masks_dir, mask_filename)

#         # --- Chargement ---
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if image is None:
#             raise FileNotFoundError(f"Image introuvable: {img_path}")

#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             raise FileNotFoundError(f"Masque introuvable: {mask_path}")

#         # --- Resize + normalisation [0,1] ---
#         image = cv2.resize(image, self.target_size).astype(np.float32) / 255.0
#         mask  = cv2.resize(mask,  self.target_size).astype(np.float32) / 255.0

#         image = torch.from_numpy(image).float().unsqueeze(0)  # [1,H,W]
#         mask  = torch.from_numpy(mask).float().unsqueeze(0)   # [1,H,W]

#         # --- Transformations synchronisées ---
#         if self.transform:
#             seed = random.randint(0, 2**32 - 1)
#             random.seed(seed); torch.manual_seed(seed)
#             image = self.transform(image)

#             random.seed(seed); torch.manual_seed(seed)
#             mask_transform = transforms.Compose([
#                 t for t in self.transform.transforms
#                 if not isinstance(t, transforms.Normalize)
#             ])
#             mask = mask_transform(mask)

#         return image, mask

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
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        pred_flat, target_flat = pred.view(-1), target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_split_indices(client_dir, N):
    """
    Split déterministe ~80/15/5 (train/val/test) par client.
    """
    g = torch.Generator().manual_seed(abs(hash(client_dir)) % (2**31))
    perm = torch.randperm(N, generator=g).tolist()

    n_test = int(round(TEST_RATIO * N))
    n_val  = int(round(VAL_RATIO  * N))
    n_train = N - n_val - n_test

    if N >= 20:
        n_test = max(1, n_test)
    else:
        n_test = 0

    if N >= 10:
        n_val = max(1, n_val)
    else:
        n_val = 0

    n_train = max(0, N - n_val - n_test)

    if n_train + n_val + n_test != N:
        diff = N - (n_train + n_val + n_test)
        n_train += diff

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ====== Helpers éval améliorée (seuil optimisé, TTA, postproc) ======
import torch.nn.functional as F

@torch.no_grad()
def tta_predict_probs(model, imgs, tta=True):
    """Retourne des proba [0..1] avec flips H/V (moyennées) si tta=True."""
    model.eval()
    logits = model(imgs)
    probs = torch.sigmoid(logits)
    if not tta:
        return probs
    probs_h = torch.sigmoid(model(torch.flip(imgs, dims=[-1])))
    probs_v = torch.sigmoid(model(torch.flip(imgs, dims=[-2])))
    probs_h = torch.flip(probs_h, dims=[-1])
    probs_v = torch.flip(probs_v, dims=[-2])
    return (probs + probs_h + probs_v) / 3.0

def postprocess_mask(bin_mask_np):
    """Fermeture morpho + plus grande composante connexe."""
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(bin_mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((closed>0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return closed
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)

@torch.no_grad()
def best_threshold_on_val(model, loader, device, tta=True, postproc=True):
    """Balaye des seuils et renvoie (best_thr, best_dice) sur la VAL."""
    model.eval()
    thresholds = [i/100 for i in range(30, 71, 5)]  # 0.30 -> 0.70
    best_thr, best_d = 0.5, -1.0
    for th in thresholds:
        d_sum, n = 0.0, 0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = (masks.to(device) > 0.5).float()
            probs = tta_predict_probs(model, imgs, tta=tta)
            preds = (probs > th).float()
            if postproc:
                preds_np = preds.squeeze(1).cpu().numpy()
                pp = []
                for p in preds_np:
                    m = (p*255).astype(np.uint8)
                    m = postprocess_mask(m)
                    pp.append(torch.from_numpy((m>0).astype(np.float32)))
                preds = torch.stack(pp, dim=0).unsqueeze(1).to(device)
            d_sum += dice_score(preds, masks).item()
            n += 1
        mean_d = d_sum / max(1, n)
        if mean_d > best_d:
            best_d, best_thr = mean_d, th
    return best_thr, best_d

@torch.no_grad()
def evaluate_tuned(model, loader, device, thr, tta=True, postproc=True):
    """Évalue avec seuil fixé + TTA + postproc. Retourne Dice moyen."""
    model.eval()
    d_sum, n = 0.0, 0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = (masks.to(device) > 0.5).float()
        probs = tta_predict_probs(model, imgs, tta=tta)
        preds = (probs > thr).float()
        if postproc:
            preds_np = preds.squeeze(1).cpu().numpy()
            pp = []
            for p in preds_np:
                m = (p*255).astype(np.uint8)
                m = postprocess_mask(m)
                pp.append(torch.from_numpy((m>0).astype(np.float32)))
            preds = torch.stack(pp, dim=0).unsqueeze(1).to(device)
        d_sum += dice_score(preds, masks).item()
        n += 1
    return d_sum / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Évalue (val/test) 'standard' sans DP/grad. Retourne (loss_moy, dice_moy)."""
    model.eval()
    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        masks = (masks > 0.5).float()  # binarise
        logits = model(imgs)
        loss = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        d = dice_score(probs, masks)
        total_loss += loss.item()
        total_dice += d.item()
        n_batches += 1
    if n_batches == 0:
        return float('nan'), float('nan')
    return total_loss / n_batches, total_dice / n_batches

# === VISUALISATION DE PRÉDICTIONS ===
def visualize_predictions(model, dataset, device, num_samples=4, save_path='sample_predictions.png'):
    import numpy as np
    model.eval()
    n = len(dataset)
    if n == 0:
        print("⚠️ visualize_predictions: dataset vide, rien à afficher.")
        return
    num_samples = min(num_samples, n)
    indices = np.random.choice(n, num_samples, replace=False)
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 12))
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            pred = model(image_input).cpu().squeeze().numpy()
            image_np = image.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            pred_sigmoid = torch.sigmoid(torch.from_numpy(pred)).numpy()
            axes[0, i].imshow(image_np, cmap='gray')
            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')
            axes[1, i].imshow(mask_np, cmap='gray')
            axes[1, i].set_title(f'Ground Truth {i+1}')
            axes[1, i].axis('off')
            axes[2, i].imshow(pred_sigmoid, cmap='gray')
            axes[2, i].set_title(f'Prediction {i+1}')
            axes[2, i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"🖼️  Sample predictions saved to {save_path}")
    model.train()


def make_loader(client_dir, indices, transform, batch_size, shuffle, drop_last):
    ds = FetalHCDataset(
        images_dir=os.path.join(client_dir, "imagesTr"),
        masks_dir=os.path.join(client_dir, "labelsTr"),
        transform=transform,
        target_size=TARGET_SIZE,
        indices=indices
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return ds, loader
# def make_loader(client_dir, indices, transform, batch_size, shuffle, drop_last):
#     images_dir = os.path.join(client_dir, "imagesTr")
#     masks_dir  = os.path.join(client_dir, "labelsTr")

#     ds = FetalHCDataset(images_dir, masks_dir, transform=transform, target_size=(256,256), indices=indices)

#     # --- GARDE-FOU 1 : si le split est vide, on évite de créer un DataLoader vide
#     if len(ds) == 0:
#         raise RuntimeError(f"[make_loader] Dataset vide pour {client_dir} avec indices={len(indices)}")

#     # --- GARDE-FOU 2 : garantir au moins 1 batch, même si batch_size > len(ds)
#     eff_bs = min(batch_size, len(ds))
#     if eff_bs <= 0:
#         eff_bs = 1

#     # Avec Opacus, drop_last est ignoré de toute façon; gardons False pour éviter les surprises
#     loader = DataLoader(ds, batch_size=eff_bs, shuffle=(shuffle and len(ds) > 1), drop_last=False)

#     # --- GARDE-FOU 3 : Opacus fait 1/len(loader); assurons-nous d'avoir >= 1
#     if len(loader) == 0:
#         # rebuild au pire avec bs=1
#         loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

#     return ds, loader

def train_one_client(client_dir, global_state_dict, cumulative_engine):
    model = UNet()
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    model.load_state_dict(global_state_dict, strict=True)
    model.to(device)

    # split déterministe par client
    base_ds = FetalHCDataset(
        images_dir=os.path.join(client_dir, "imagesTr"),
        masks_dir=os.path.join(client_dir, "labelsTr"),
        transform=None,
        target_size=TARGET_SIZE
    )
    N = len(base_ds)
    train_idx, val_idx, test_idx = get_split_indices(client_dir, N)

    # Loaders
    train_ds, train_loader = make_loader(client_dir, train_idx, get_train_transforms(), BATCH_SIZE, True,  True)
    val_ds,   val_loader   = make_loader(client_dir, val_idx,   None,                 BATCH_SIZE, False, False)

    # DP params basés sur TRAIN uniquement
    delta = 1 / max(1, len(train_ds))
    sample_rate = BATCH_SIZE / max(1, len(train_ds))

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=4, cooldown=2, min_lr=5e-6
    )
    criterion = BCEDiceLoss()

    if cumulative_engine is None:
        cumulative_engine = PrivacyEngine(accountant="rdp")

    model.train()
    model, optimizer, train_loader = cumulative_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
        sample_rate=sample_rate
    )

    # ⬅️ ajout: époques locales plus longues pour client1 uniquement
    cid = os.path.basename(client_dir.rstrip("/"))
    is_client1 = (cid == "client1")
    local_epochs = EPOCHS_CLIENT1 if is_client1 else EPOCHS

    model, history = train_model_client(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=local_epochs,   # ⬅️ seul changement
        device=device
    )

    # Validation (standard)
    val_loss, val_dice = evaluate(model, val_loader, criterion, device)

    # Validation 'tuned' (seuil optimisé + TTA + postproc) — reporting
    thr_val, tuned_val_dice = best_threshold_on_val(model, val_loader, device, tta=ENABLE_TTA, postproc=ENABLE_POSTPROC)
    print(f"[Client @ {os.path.basename(client_dir)}] Val Dice tuned={tuned_val_dice:.4f} @ thr={thr_val:.2f}")

    try:
        scheduler.step(val_loss)
    except Exception:
        pass

    try:
        epsilon = cumulative_engine.accountant.get_epsilon(delta=delta)
    except Exception as e:
        print(f"[CLIENT] Failed to compute epsilon: {e}")
        epsilon = 0.0

    final_loss = history['loss'][-1] if history['loss'] else 1.0
    final_dice = history['dice'][-1] if history['dice'] else 0.0

    print(
        f"[Client @ {os.path.basename(client_dir)}] "
        f"Train Loss={final_loss:.4f} | Train Dice={final_dice:.4f} | "
        f"Val Loss={val_loss:.4f} | Val Dice={val_dice:.4f} | Eps={epsilon:.4f} | "
        f"n_train={len(train_ds)} n_val={len(val_ds)} n_test={len(test_idx)}"
    )

    raw_state = model.state_dict()
    clean_state = {k[len("_module."):] if k.startswith("_module.") else k: v.detach().cpu() for k, v in raw_state.items()}

    del model, train_loader, val_loader, optimizer, scheduler, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return clean_state, len(train_ds), final_loss, final_dice, epsilon, cumulative_engine, val_loss, val_dice, (train_idx, val_idx, test_idx), len(val_ds)

# === Init global model ===
global_model = UNet().to(device)
if not ModuleValidator.is_valid(global_model):
    global_model = ModuleValidator.fix(global_model)
global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

# === Main Loop ===
cumulative_engine = None
start_time = time.time()
for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n[ROUND {rnd}/{NUM_ROUNDS}]")
    states, sizes, losses, dices, epsilons = [], [], [], [], []
    vlosses, vdices = [], []
    val_sizes = {}

    for client_dir in CLIENT_DIRS:
        (state, n, loss, dice, eps, cumulative_engine, vloss, vdice, splits, n_val) = train_one_client(
            client_dir, global_state, cumulative_engine
        )
        states.append(state); sizes.append(n)
        losses.append(loss); dices.append(dice); epsilons.append(eps)
        vlosses.append(vloss); vdices.append(vdice)

        cid = os.path.basename(client_dir.rstrip("/"))
        if cid not in client_splits:
            client_splits[cid] = splits  # (train_idx, val_idx, test_idx)

        client_train_loss[cid].append(loss)
        client_train_dice[cid].append(dice)
        client_val_loss[cid].append(vloss)
        client_val_dice[cid].append(vdice)
        val_sizes[cid] = n_val

    # --- Val Dice par client + moyenne pondérée (avant agrégation)
    print(f"[ROUND {rnd}] Val Dice par client: " +
          ", ".join(f"{cid}={client_val_dice[cid][-1]:.3f}" for cid in client_ids))
    weighted_val = np.average(
        [client_val_dice[cid][-1] for cid in client_ids],
        weights=[val_sizes[cid] for cid in client_ids]
    )
    print(f"[ROUND {rnd}] Weighted Mean Val Dice={weighted_val:.4f}")

    # Agrégation pondérée par la taille d'entraînement
    total = sum(sizes) if len(sizes) > 0 else 1
    global_state = {k: sum(state[k] * (sz / total) for state, sz in zip(states, sizes)) for k in states[0].keys()}

    # Moyennes locales (train/val) rapportées par clients
    loss_per_round.append(np.mean(losses) if losses else np.nan)
    dice_per_round.append(np.mean(dices) if dices else np.nan)
    epsilon_per_round.append(np.mean(epsilons) if epsilons else np.nan)
    val_loss_per_round.append(np.nanmean(vlosses) if vlosses else np.nan)
    val_dice_per_round.append(np.nanmean(vdices) if vdices else np.nan)

    print(f"[ROUND {rnd}] Mean Val Loss={val_loss_per_round[-1]:.4f} | Mean Val Dice={val_dice_per_round[-1]:.4f}")

    # === GLOBAL model on VAL (standard + tuned) ===
    with torch.no_grad():
        global_model_eval = UNet().to(device)
        if not ModuleValidator.is_valid(global_model_eval):
            global_model_eval = ModuleValidator.fix(global_model_eval)
        global_model_eval.load_state_dict(global_state, strict=True)
        global_model_eval.eval()

        g_val_losses, g_val_dices = [], []
        g_val_dices_tuned, g_val_thrs = [], []
        for client_dir in CLIENT_DIRS:
            cid = os.path.basename(client_dir.rstrip("/"))
            _, val_idx, _ = client_splits[cid]
            images_dir = os.path.join(client_dir, "imagesTr")
            masks_dir  = os.path.join(client_dir, "labelsTr")
            val_ds = FetalHCDataset(images_dir, masks_dir, transform=None, target_size=TARGET_SIZE, indices=val_idx)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

            vloss, vdice = evaluate(global_model_eval, val_loader, BCEDiceLoss(), device)
            g_val_losses.append(vloss); g_val_dices.append(vdice)

            thr_c, tuned_c = best_threshold_on_val(global_model_eval, val_loader, device, tta=ENABLE_TTA, postproc=ENABLE_POSTPROC)
            g_val_dices_tuned.append(tuned_c); g_val_thrs.append(thr_c)

        mean_global_val_loss = float(np.nanmean(g_val_losses)) if g_val_losses else float('nan')
        mean_global_val_dice = float(np.nanmean(g_val_dices)) if g_val_dices else float('nan')
        mean_global_val_dice_tuned = float(np.nanmean(g_val_dices_tuned)) if g_val_dices_tuned else float('nan')

        print(f"[ROUND {rnd}] GLOBAL on VAL -> Loss={mean_global_val_loss:.4f} | Dice={mean_global_val_dice:.4f}")
        print(f"[ROUND {rnd}] GLOBAL on VAL (tuned+TTA+PP) -> Dice={mean_global_val_dice_tuned:.4f} | thrs=" +
              ", ".join(f"{t:.2f}" for t in g_val_thrs))

        # --- Early stop sur la métrique choisie (tuned ou standard) ---
        # --- Suivi du "meilleur" modèle (sans early stopping) ---
        metric_for_es = mean_global_val_dice_tuned if USE_TUNED_FOR_EARLYSTOP else mean_global_val_dice
        if not np.isnan(metric_for_es) and metric_for_es > best_global_val_dice + 1e-6:
            best_global_val_dice = metric_for_es
            best_global_state = {k: v.clone().cpu() for k, v in global_state.items()}
            best_round = rnd


end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"\n=== Entraînement terminé en {int(minutes)} min {int(seconds)} sec ===")

# Sauvegarde du meilleur modèle global (selon métrique choisie)
if best_global_state is not None:
    torch.save(best_global_state, "best_global_state_multicenter.pth")
    print(f"💾 Best GLOBAL sauvegardé: round={best_round} | Global Val Dice*={best_global_val_dice:.4f} -> best_global_state.pth")

# === Dice final (moyenne des 3 clients, dernier round, train/val standard) ===
final_train_dice = dice_per_round[-1] if dice_per_round else float('nan')
final_val_dice   = val_dice_per_round[-1] if val_dice_per_round else float('nan')
print(f"\n🎯 Dice final (moyenne des 3 clients, dernier round) :")
print(f"   - Train Dice : {final_train_dice:.4f}")
print(f"   - Val   Dice : {final_val_dice:.4f}")

# === TEST FINAL : évalue le 'dernier' global ET le 'best_global_state' (utile pour TFE) ===
def eval_on_test(global_state_dict, label):
    test_losses, test_dices, test_accs = [], [], []
    criterion = BCEDiceLoss()
    with torch.no_grad():
        test_model = UNet().to(device)
        if not ModuleValidator.is_valid(test_model):
            test_model = ModuleValidator.fix(test_model)
        test_model.load_state_dict(global_state_dict, strict=True)
        test_model.eval()
        for client_dir in CLIENT_DIRS:
            cid = os.path.basename(client_dir.rstrip("/"))
            _, _, test_idx = client_splits[cid]
            test_ds = FetalHCDataset(
                images_dir=os.path.join(client_dir, "imagesTr"),
                masks_dir=os.path.join(client_dir, "labelsTr"),
                transform=None,
                target_size=TARGET_SIZE,
                indices=test_idx
            )
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
            
            total_loss, total_dice, total_acc, n_batches = 0.0, 0.0, 0.0, 0
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                masks_bin = (masks > 0.5).float()
                logits = test_model(imgs)
                loss = criterion(logits, masks_bin)
                probs = torch.sigmoid(logits)
                total_loss += loss.item()
                total_dice += dice_score(probs, masks_bin).item()
                total_acc  += pixel_accuracy(probs, masks_bin).item()
                n_batches += 1
            
            if n_batches > 0:
                avg_loss = total_loss / n_batches
                avg_dice = total_dice / n_batches
                avg_acc  = total_acc / n_batches
            else:
                avg_loss, avg_dice, avg_acc = float('nan'), float('nan'), float('nan')

            test_losses.append(avg_loss)
            test_dices.append(avg_dice)
            test_accs.append(avg_acc)

            print(f"[TEST {label}] {cid}: Loss={avg_loss:.4f} | Dice={avg_dice:.4f} | Acc={avg_acc:.4f} | n_test={len(test_ds)}")

    print(f"🧪 [{label}] Moyenne Test Loss: {float(np.nanmean(test_losses)):.4f} | "
          f"Moyenne Test Dice: {float(np.nanmean(test_dices)):.4f} | "
          f"Moyenne Test Acc: {float(np.nanmean(test_accs)):.4f}")

# Dernier global
eval_on_test(global_state, "last")
# Meilleur global (selon Val Dice*)
if best_global_state is not None:
    eval_on_test(best_global_state, "best")

# === Courbes (globales) ===
fig, ax1 = plt.subplots()
ax1.set_xlabel("Round")
ax1.set_ylabel("Loss / Dice")
rounds = range(1, len(loss_per_round) + 1)
ax1.plot(rounds, loss_per_round, label="Train Loss", color="blue")
ax1.plot(rounds, dice_per_round, label="Train Dice", color="green")
ax1.plot(rounds, val_loss_per_round, label="Val Loss", color="blue", linestyle="--")
ax1.plot(rounds, val_dice_per_round, label="Val Dice", color="green", linestyle="--")
ax1.tick_params(axis='y')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel("Epsilon")
ax2.plot(rounds, epsilon_per_round, label="Epsilon", color="orange")
ax2.tick_params(axis='y')

plt.title(f"Loss, Dice (Train/Val) & Epsilon vs Rounds (noise={NOISE_MULTIPLIER})")
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", ncol=3)
plt.tight_layout()
plt.savefig(f"metrics_combined_noise_{NOISE_MULTIPLIER}_multicenter.png")
plt.close()

# === Courbes par client ===
def plot_per_client(metric_dict, title, ylabel, outfile):
    plt.figure()
    for cid in client_ids:
        y = metric_dict[cid]
        x = range(1, len(y) + 1)
        plt.plot(x, y, label=cid)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

plot_per_client(client_train_dice, f"Train Dice par client vs Rounds (noise={NOISE_MULTIPLIER}) multicenter", "Dice", f"per_client_train_dice_{NOISE_MULTIPLIER}_multicenter.png")
plot_per_client(client_train_loss, f"Train Loss par client vs Rounds (noise={NOISE_MULTIPLIER}) multicenter", "Loss", f"per_client_train_loss_{NOISE_MULTIPLIER}_multicenter.png")
plot_per_client(client_val_dice,  f"Val Dice par client vs Rounds (noise={NOISE_MULTIPLIER}) multicenter",   "Dice", f"per_client_val_dice_{NOISE_MULTIPLIER}_multicenter.png")
plot_per_client(client_val_loss,  f"Val Loss par client vs Rounds (noise={NOISE_MULTIPLIER}) multicenter",   "Loss", f"per_client_val_loss_{NOISE_MULTIPLIER}_multicenter.png")

print("\n✅ Graphe global + graphes par client générés.")
print("✅ Éval 'tuned' (seuil + TTA + postproc) activée pour la validation.")

# === VISUALISATION FINALE DE PRÉDICTIONS (après le dernier round) ===
def _build_viz_dataset_from_splits(use_test=True):
    """Concatène les datasets (test si dispo, sinon val) de chaque client pour la visualisation."""
    subsets = []
    for client_dir in CLIENT_DIRS:
        cid = os.path.basename(client_dir.rstrip("/"))
        if cid not in client_splits:
            continue
        train_idx, val_idx, test_idx = client_splits[cid]
        idx = test_idx if (use_test and len(test_idx) > 0) else val_idx
        if len(idx) == 0:
            continue
        ds = FetalHCDataset(
            images_dir=os.path.join(client_dir, "imagesTr"),
            masks_dir=os.path.join(client_dir, "labelsTr"),
            transform=None,
            target_size=TARGET_SIZE,
            indices=idx
        )
        subsets.append(ds)
    if len(subsets) == 0 and use_test:
        # fallback: retente avec val si aucun test
        return _build_viz_dataset_from_splits(use_test=False)
    if len(subsets) == 0:
        return None
    return subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)

viz_state = best_global_state if best_global_state is not None else global_state
viz_model = UNet().to(device)
if not ModuleValidator.is_valid(viz_model):
    viz_model = ModuleValidator.fix(viz_model)
viz_model.load_state_dict(viz_state, strict=True)
viz_model.eval()

viz_dataset = _build_viz_dataset_from_splits(use_test=True)
if viz_dataset is not None:
    num_samples = 4  # ajuste si tu veux
    save_path = f"sample_predictions_noise_{NOISE_MULTIPLIER}_multicenter.png"
    visualize_predictions(viz_model, viz_dataset, device, num_samples=num_samples, save_path=save_path)
else:
    print("⚠️ Aucun dataset (test/val) disponible pour la visualisation de prédictions.")
