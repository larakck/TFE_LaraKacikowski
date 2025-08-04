# === utils/dataset.py ===

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch

class ClientDataset(Dataset):
    def __init__(self, data_dir, split="train", seed=42, img_size=128, augment=False):
        self.img_size = img_size

        self.samples = []
        all_samples = []
        images_dir = os.path.join(data_dir, "imagesTr")
        labels_dir = os.path.join(data_dir, "labelsTr")

        for img_name, lab_name in zip(sorted(os.listdir(images_dir)), sorted(os.listdir(labels_dir))):
            all_samples.append((os.path.join(images_dir, img_name), os.path.join(labels_dir, lab_name)))

        train, temp = train_test_split(all_samples, test_size=0.3, random_state=seed)
        val, _ = train_test_split(temp, test_size=0.5, random_state=seed)

        self.samples = train if split == "train" else val

        # === Transforms ===
        base_transforms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
        ]

        if augment and split == "train":
            print("[Dataset] Augmentation activée pour ce client")
            base_transforms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
                #transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),#amodif 
                transforms.GaussianBlur(kernel_size=3)
            ]

        base_transforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisation ajoutée
        ]

        self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)

        # Resize + binarisation du masque
        mask = transforms.Resize(
            (self.img_size, self.img_size),
                interpolation=transforms.InterpolationMode.NEAREST  # ⬅️ très important
        )(mask)




        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.1).float()
        #print("Mask unique values:", torch.unique(mask))


        return image, mask

def get_dataloaders(data_path, batch_size=8, img_size=128, augment=False):
    train_ds = ClientDataset(data_path, "train", img_size=img_size, augment=augment)
    val_ds = ClientDataset(data_path, "val", img_size=img_size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    return train_dl, val_dl
