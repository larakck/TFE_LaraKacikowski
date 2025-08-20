import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

import os, re, cv2, random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FetalHCDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256), mask_suffix=None):
        # On prend les dossiers tels quels (imagesTr / labelsTr)
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform = transform
        self.target_size = target_size

        # Précharge la liste des fichiers masques pour lookup O(1)
        self._mask_files = set(f for f in os.listdir(self.masks_dir) if f.endswith(".png"))

        # On ne se fie plus à mask_suffix pour décider du pattern : on s'en sert juste en "priorité"
        self.mask_suffix = mask_suffix  # peut être None, "_gt_", "_mask.png", "_Annotation.png", etc.

        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".png")])
        if not self.image_files:
            raise FileNotFoundError(f"Aucune image .png dans {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def _resolve_mask_name(self, img_name: str) -> str:
        """Choisit le bon masque pour cette image en testant tous les patterns connus."""
        base = img_name.split('_')[0]  # ex "003" pour "003_0000.png"

        # 1) Sierra Leone candidats
        gt_candidates = [
            f"filled_{base}_gt_{base}.png",
            f"{base}_gt_{base}.png",
        ]

        # 2) Egypt/Algeria : ..._0000.png -> ..._mask.png
        mask_candidate = re.sub(r"_\d{4}\.png$", "_mask.png", img_name)
        if mask_candidate == img_name:
            mask_candidate = img_name.replace(".png", "_mask.png")

        # 3) HC18 : ... -> ..._Annotation.png
        annot_candidate = img_name.replace(".png", "_Annotation.png")

        # Ordre de priorité :
        # - si mask_suffix est donné, on le teste d’abord
        ordered = []
        if self.mask_suffix == "_gt_":
            ordered.extend(gt_candidates)
        elif self.mask_suffix == "_mask.png":
            ordered.append(mask_candidate)
        elif self.mask_suffix == "_Annotation.png":
            ordered.append(annot_candidate)

        # Ajoute ensuite tous les patterns, sans doublon
        for cand in [*gt_candidates, mask_candidate, annot_candidate]:
            if cand not in ordered:
                ordered.append(cand)

        # Sélectionne le premier qui existe réellement
        for cand in ordered:
            if cand in self._mask_files:
                return cand

        # Rien trouvé -> message d’erreur très parlant
        samples = list(self._mask_files)[:5]
        raise FileNotFoundError(
            f"[Mask introuvable] Aucun parmi {ordered} dans {self.masks_dir}. "
            f"Exemples présents: {samples}"
        )

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"[Image introuvable] {img_path}")
        image = cv2.resize(image, self.target_size) / 255.0

        mask_name = self._resolve_mask_name(img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"[Mask illisible] {mask_path}")
        mask = cv2.resize(mask, self.target_size) / 255.0

        image = torch.from_numpy(image).float().unsqueeze(0)
        mask  = torch.from_numpy(mask ).float().unsqueeze(0)

        if self.transform:
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed); torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed); torch.manual_seed(seed)
            mask_transform = transforms.Compose(
                [t for t in self.transform.transforms if not isinstance(t, transforms.Normalize)]
            )
            mask = mask_transform(mask)

        return image, mask



# import os
# import torch
# import numpy as np
# import cv2
# import random
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms

# class FetalHCDataset(Dataset):
#     """
#     Dataset adapté pour 1327317/training_set_processed avec imagesTr et labelsTr.
#     Les masques sont nommés : <nom_image>_Annotation.png
#     """
#     def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256)):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.transform = transform
#         self.target_size = target_size

#         self.image_files = sorted([
#             f for f in os.listdir(images_dir)
#             if f.endswith('.png') and '_Annotation' not in f
#         ])

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_filename = self.image_files[idx]
#         mask_filename = img_filename.replace('.png', '_Annotation.png')

#         img_path = os.path.join(self.images_dir, img_filename)
#         mask_path = os.path.join(self.masks_dir, mask_filename)

#         # Lecture de l'image en niveau de gris
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if image is None:
#             raise FileNotFoundError(f"Image not found: {img_path}")
#         image = cv2.resize(image, self.target_size)
#         image = image / 255.0

#         # Lecture du masque
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             raise FileNotFoundError(f"Mask not found: {mask_path}")
#         mask = cv2.resize(mask, self.target_size)
#         mask = mask / 255.0

#         image = torch.from_numpy(image).float().unsqueeze(0)
#         mask = torch.from_numpy(mask).float().unsqueeze(0)

#         if self.transform:
#             seed = random.randint(0, 2**32)
#             random.seed(seed)
#             torch.manual_seed(seed)
#             image = self.transform(image)

#             random.seed(seed)
#             torch.manual_seed(seed)
#             mask_transform = transforms.Compose([
#                 t for t in self.transform.transforms if not isinstance(t, transforms.Normalize)
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

