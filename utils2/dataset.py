# import os
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# import random

# class FetalHCDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256), mask_suffix='_mask.png'):
#         if os.path.exists(os.path.join(images_dir, 'images')):
#             self.images_dir = os.path.join(images_dir, 'images')
#             self.masks_dir = os.path.join(masks_dir, 'images')
#         else:
#             self.images_dir = images_dir
#             self.masks_dir = masks_dir

#         self.transform = transform
#         self.target_size = target_size
#         self.mask_suffix = mask_suffix
#         self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         image = cv2.resize(image, self.target_size) if image is not None else np.zeros(self.target_size)
#         image = image / 255.0

#         if self.mask_suffix == "_gt_":
#             base_num = img_name.split('_')[0]
#             mask_name = f"{base_num}_gt_{base_num}.png"
#         else:
#             mask_name = img_name.replace('.png', self.mask_suffix)

#         mask_path = os.path.join(self.masks_dir, mask_name)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask = cv2.resize(mask, self.target_size) if mask is not None else np.zeros(self.target_size)
#         mask = mask / 255.0

#         image = torch.from_numpy(image).float().unsqueeze(0)
#         mask = torch.from_numpy(mask).float().unsqueeze(0)

#         if self.transform:
#             seed = random.randint(0, 2**32)
#             random.seed(seed); torch.manual_seed(seed)
#             image = self.transform(image)
#             random.seed(seed); torch.manual_seed(seed)
#             mask_transform = transforms.Compose([
#                 t for t in self.transform.transforms if not isinstance(t, transforms.Normalize)
#             ])
#             mask = mask_transform(mask)

#         return image, mask



import os
import torch
import numpy as np
import cv2
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FetalHCDataset(Dataset):
    """
    Dataset adapté pour 1327317/training_set_processed avec imagesTr et labelsTr.
    Les masques sont nommés : <nom_image>_Annotation.png
    """
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

        # Lecture de l'image en niveau de gris
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.resize(image, self.target_size)
        image = image / 255.0

        # Lecture du masque
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = cv2.resize(mask, self.target_size)
        mask = mask / 255.0

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

