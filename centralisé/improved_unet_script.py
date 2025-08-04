import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F_torch
from torchvision.transforms import functional as F
from math import pi
import glob
import re
from PIL import Image
import random
import matplotlib.pyplot as plt

# --- OPACUS CHANGE START ---
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
# --- OPACUS CHANGE END ---

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Loss Functions and Metrics ---
class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss for better segmentation performance.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target):
        """Calculate Dice loss from logits"""
        pred_sigmoid = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # Return dice loss (1 - dice coefficient)
        return 1 - dice_coeff
    
    def forward(self, pred, target):
        # BCE loss
        bce = self.bce_loss(pred, target)
        
        # Dice loss
        dice = self.dice_loss(pred, target)
        
        # Combined loss
        combined_loss = self.bce_weight * bce + self.dice_weight * dice
        return combined_loss

def dice_score(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient for binary segmentation with NaN protection.
    """
    # Apply sigmoid if needed (for logits)
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    # Clamp to avoid numerical issues
    dice = torch.clamp(dice, 0.0, 1.0)
    return dice

def pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy with NaN protection.
    """
    # Apply sigmoid if needed (for logits)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    correct = (pred == target).sum()
    total = target.numel()
    
    if total == 0:
        return torch.tensor(0.0)
    
    accuracy = correct / total
    return torch.clamp(accuracy, 0.0, 1.0)

def iou_score(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) score with NaN protection.
    """
    # Apply sigmoid if needed (for logits)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return torch.tensor(1.0 if intersection == 0 else 0.0)
    
    iou = (intersection + smooth) / (union + smooth)
    return torch.clamp(iou, 0.0, 1.0)

def plot_training_metrics(history, save_path='training_metrics.png'):
    """
    Plot training metrics (loss, accuracy, dice score, IoU).
    """
    epochs = range(1, len(history['loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['accuracy'], 'g-', label='Training Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Dice score plot
    ax3.plot(epochs, history['dice'], 'r-', label='Training Dice Score')
    ax3.set_title('Training Dice Score')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Dice Score')
    ax3.legend()
    ax3.grid(True)
    
    # IoU score plot
    ax4.plot(epochs, history['iou'], 'm-', label='Training IoU Score')
    ax4.set_title('Training IoU Score')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('IoU Score')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training metrics plot saved to {save_path}")

def visualize_predictions(model, dataset, device, num_samples=4, save_path='sample_predictions.png'):
    """
    Visualize sample predictions from the model.
    """
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 12))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            
            # Get prediction
            pred = model(image_input)
            pred = pred.cpu().squeeze().numpy()
            
            # Convert to numpy for visualization
            image_np = image.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            
            # Plot original image
            axes[0, i].imshow(image_np, cmap='gray')
            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')
            
            # Plot ground truth mask
            axes[1, i].imshow(mask_np, cmap='gray')
            axes[1, i].set_title(f'Ground Truth {i+1}')
            axes[1, i].axis('off')
            
            # Plot prediction (apply sigmoid for visualization since model outputs logits)
            pred_sigmoid = torch.sigmoid(torch.from_numpy(pred)).numpy()
            axes[2, i].imshow(pred_sigmoid, cmap='gray')
            axes[2, i].set_title(f'Prediction {i+1}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Sample predictions saved to {save_path}")
    
    model.train()  # Set back to training mode

# --- 2. U-Net Model Architecture (HC18 dataset compatible) ---
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(4, out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(4, out_ch),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        self.enc1 = CBR(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = CBR(512, 1024)
        
        # Replace ConvTranspose2d with Upsample + Conv2d for Opacus compatibility
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv4 = nn.Conv2d(1024, 512, kernel_size=1)
        self.dec4 = CBR(1024, 512)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.dec3 = CBR(512, 256)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.dec2 = CBR(256, 128)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv1 = nn.Conv2d(128, 64, kernel_size=1)
        self.dec1 = CBR(128, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with upsampling
        up4 = self.up4(bottleneck)
        up4 = self.upconv4(up4)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        
        up3 = self.up3(dec4)
        up3 = self.upconv3(up3)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        
        up2 = self.up2(dec3)
        up2 = self.upconv2(up2)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        
        up1 = self.up1(dec2)
        up1 = self.upconv1(up1)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        return self.final(dec1)

# --- 3. Custom Dataset Class (from main_dp_opacus.py) ---
class FetalHCDataset(Dataset):
    """
    Custom PyTorch Dataset for fetal head circumference segmentation.
    """
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256), mask_suffix='_mask.png'):
        # Check if we have the organized structure (with 'images' subdirectory) or direct structure
        if os.path.exists(os.path.join(images_dir, 'images')):
            self.images_dir = os.path.join(images_dir, 'images')
            self.masks_dir = os.path.join(masks_dir, 'images')
        else:
            # Direct structure (augmented processed data)
            self.images_dir = images_dir
            self.masks_dir = masks_dir
        
        self.transform = transform
        self.target_size = target_size
        self.mask_suffix = mask_suffix
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        self.image_files.sort()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            # Return a black image if loading fails
            image = np.zeros(self.target_size, dtype=np.uint8)
        else:
            image = cv2.resize(image, self.target_size)
        
        image = image / 255.0  # Normalize to [0, 1]
        
        # Load mask - handle different naming conventions
        if self.mask_suffix == "_gt_":
            # Sierra Leone dataset: 003_0000.png -> 003_gt_003.png
            base_num = img_name.split('_')[0]  # Extract '003' from '003_0000.png'
            mask_name = f"{base_num}_gt_{base_num}.png"
        else:
            # Standard naming: append suffix
            mask_name = img_name.replace('.png', self.mask_suffix)
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Could not load mask {mask_path}")
            # Return a black mask if loading fails
            mask = np.zeros(self.target_size, dtype=np.uint8)
        else:
            mask = cv2.resize(mask, self.target_size)
        
        mask = mask / 255.0  # Normalize to [0, 1]
        
        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # Add channel dimension
        
        # Apply transforms if provided
        if self.transform:
            # For data augmentation, we need to apply the same transform to both image and mask
            seed = random.randint(0, 2**32)
            
            # Apply to image
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            
            # Apply to mask (without normalization)
            random.seed(seed)
            torch.manual_seed(seed)
            mask_transform = transforms.Compose([
                t for t in self.transform.transforms 
                if not isinstance(t, transforms.Normalize)
            ])
            mask = mask_transform(mask)
        
        return image, mask

# --- 4. Data Augmentation ---
def get_train_transforms():
    """
    Get training data augmentation transforms optimized for medical segmentation.
    """
    return transforms.Compose([
        transforms.RandomRotation(15),  # Slightly more rotation for better generalization
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),  # Add vertical flip for ultrasound images
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5  # Small shear for better robustness
        ),
        # Add elastic transform would be ideal but requires more complex implementation
    ])

# --- 5. Training Pipeline ---
def train_model(images_path, masks_path, model_save_path='unet_hc18_multicenter.pth', epochs=10, batch_size=4, learning_rate=1e-3, mask_suffix='_mask.png',
                enable_dp=True, noise_multiplier=1.0, max_grad_norm=1.0):
    """
    Train the multicenter U-Net model on the HC18 data.
    """
    print("Building multicenter U-Net model...")
    model = UNet()

    # --- OPACUS CHANGE START ---
    # It's best practice to validate the model is compatible with Opacus.
    # The fix method can replace some common incompatible layers (like BatchNorm).
    if not ModuleValidator.is_valid(model):
        print("Model contains incompatible layers. Attempting to fix...")
        model = ModuleValidator.fix(model)
    
    model = model.to(device)
    # --- OPACUS CHANGE END ---
    
    # Loss function and optimizer - Using combined BCE-Dice loss for better segmentation
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add DP-optimized learning rate scheduler
    if enable_dp:
        # More conservative scheduler for DP training
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,        # Less aggressive reduction for DP
            patience=8,        # More patient for noisy DP gradients
            min_lr=1e-5,       # Higher minimum LR for DP
            cooldown=3
        )
        print("Using DP-optimized learning rate scheduler")
    else:
        # Standard scheduler for non-DP training
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=5,
            min_lr=5e-6,
            cooldown=2
        )
        print("Using standard learning rate scheduler")
    
    print("Setting up data loaders...")
    
    # Create dataset and data loader
    train_dataset = FetalHCDataset(
        images_path, 
        masks_path, 
        transform=get_train_transforms(),
        target_size=(256, 256),
        mask_suffix=mask_suffix
    )
    
    # --- OPACUS CHANGE START ---
    privacy_engine = None
    target_delta = None
    
    if enable_dp:
        # The `target_delta` should be smaller than 1/len(dataset)
        target_delta = 1 / len(train_dataset)
        print(f"Setting DP target delta to {target_delta:.2e}")
        # Opacus requires the DataLoader to drop the last batch if it's smaller than the rest
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            drop_last=True  # Important for Opacus
        )
        
        # Attach the PrivacyEngine to the model, optimizer, and dataloader
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        print(f"Attached Opacus PrivacyEngine with noise multiplier {noise_multiplier} and max_grad_norm {max_grad_norm}")
    else:
        # Regular training without DP
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        print("Training without differential privacy")
    # --- OPACUS CHANGE END ---
    
    print(f"Training on {len(train_dataset)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Training loop
    model.train()
    training_history = {'loss': [], 'accuracy': [], 'dice': [], 'iou': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        running_dice = 0.0
        running_iou = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # The optimizer step and gradient clipping are handled by the PrivacyEngine's hook
            optimizer.step()
            
            # Calculate metrics with NaN protection
            batch_accuracy = pixel_accuracy(outputs, masks)
            batch_dice = dice_score(outputs, masks)
            batch_iou = iou_score(outputs, masks)
            
            # Check for NaN values and skip if found
            if torch.isnan(loss) or torch.isnan(batch_accuracy) or torch.isnan(batch_dice) or torch.isnan(batch_iou):
                print(f"Warning: NaN detected in batch {batch_idx}, skipping...")
                continue
                
            # Also check for infinite values
            if torch.isinf(loss) or torch.isinf(batch_accuracy) or torch.isinf(batch_dice) or torch.isinf(batch_iou):
                print(f"Warning: Inf detected in batch {batch_idx}, skipping...")
                continue
            
            # Statistics
            running_loss += loss.item()
            running_accuracy += batch_accuracy.item()
            running_dice += batch_dice.item()
            running_iou += batch_iou.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {batch_accuracy.item():.4f}, Dice: {batch_dice.item():.4f}, IoU: {batch_iou.item():.4f}')
        
        # Calculate epoch metrics with protection against division by zero
        if num_batches > 0:
            epoch_loss = running_loss / num_batches
            epoch_accuracy = running_accuracy / num_batches
            epoch_dice = running_dice / num_batches
            epoch_iou = running_iou / num_batches
        else:
            print(f"Warning: No valid batches in epoch {epoch+1}, using previous epoch values")
            if epoch > 0:
                epoch_loss = training_history['loss'][-1]
                epoch_accuracy = training_history['accuracy'][-1]
                epoch_dice = training_history['dice'][-1]
                epoch_iou = training_history['iou'][-1]
            else:
                epoch_loss = 1.0
                epoch_accuracy = 0.0
                epoch_dice = 0.0
                epoch_iou = 0.0
        
        training_history['loss'].append(epoch_loss)
        training_history['accuracy'].append(epoch_accuracy)
        training_history['dice'].append(epoch_dice)
        training_history['iou'].append(epoch_iou)
        
        # Update learning rate scheduler
        scheduler.step(epoch_loss)
        
        # --- OPACUS CHANGE START ---
        if enable_dp:
            # Get the privacy budget (epsilon) for the current state
            epsilon = privacy_engine.get_epsilon(delta=target_delta)
            print(f'Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}, (ε = {epsilon:.2f}, δ = {target_delta:.2e})')
        else:
            print(f'Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        # --- OPACUS CHANGE END ---
    
    print(f"Training complete. Saving model to {model_save_path}")
    
    # --- OPACUS CHANGE START ---
    if enable_dp:
        # It's important to save the non-private version of the model.
        torch.save(model._module.state_dict(), model_save_path)
    else:
        # Regular model saving without DP hooks
        torch.save(model.state_dict(), model_save_path)
    # --- OPACUS CHANGE END ---
    
    # Print training summary
    print("\n=== Training Summary ===")
    print(f"Best Loss: {min(training_history['loss']):.4f}")
    print(f"Best Accuracy: {max(training_history['accuracy']):.4f}")
    print(f"Best Dice Score: {max(training_history['dice']):.4f}")
    print(f"Best IoU Score: {max(training_history['iou']):.4f}")
    print(f"Final Loss: {training_history['loss'][-1]:.4f}")
    print(f"Final Accuracy: {training_history['accuracy'][-1]:.4f}")
    print(f"Final Dice Score: {training_history['dice'][-1]:.4f}")
    print(f"Final IoU Score: {training_history['iou'][-1]:.4f}")
    
    return model, training_history

# --- 6. Main Execution ---
if __name__ == '__main__':
    # Choose your dataset - uncomment one of the following:
    BASE_DATA_DIR = 'multicenter/external/Dataset004_SierraLeone'  # Multicenter dataset
    # BASE_DATA_DIR = './1327317'  # HC18 dataset
    
    # Detect dataset type and configure paths accordingly
    if 'multicenter' in BASE_DATA_DIR or 'Dataset' in BASE_DATA_DIR:
        # Multicenter dataset configuration
        dataset_type = "hc18_multicenter_dp"
        
        # Check if this is the Sierra Leone dataset structure (imagesTr/labelsTr_trunctated)
        if os.path.exists(os.path.join(BASE_DATA_DIR, 'imagesTr')):
            IMAGES_DIR = os.path.join(BASE_DATA_DIR, 'imagesTr')
            # Use truncated masks if available, otherwise use filled masks
            if os.path.exists(os.path.join(BASE_DATA_DIR, 'labelsTr_trunctated')):
                MASKS_DIR = os.path.join(BASE_DATA_DIR, 'labelsTr_trunctated')
                mask_suffix = "_gt_"  # Custom mask suffix for Sierra Leone truncated format
                print("Using truncated masks for Sierra Leone dataset")
            else:
                MASKS_DIR = os.path.join(BASE_DATA_DIR, 'labelsTr')
                mask_suffix = "_gt_"  # Custom mask suffix for Sierra Leone filled format
                print("Using filled masks for Sierra Leone dataset")
        # Check for organized training data structure
        elif os.path.exists(os.path.join(BASE_DATA_DIR, 'organized_training_data')):
            # Use truncated images if available, otherwise use regular images
            truncated_images_dir = os.path.join(BASE_DATA_DIR, 'organized_training_data', 'images_truncated')
            if os.path.exists(truncated_images_dir):
                IMAGES_DIR = truncated_images_dir
                print("Using truncated images for multicenter dataset")
            else:
                IMAGES_DIR = os.path.join(BASE_DATA_DIR, 'organized_training_data', 'images')
                print("Truncated images not found, using regular images")
            
            MASKS_DIR = os.path.join(BASE_DATA_DIR, 'organized_training_data', 'masks')
            mask_suffix = "_Annotation.png"
        else:
            print(f"Error: Could not find expected directory structure in {BASE_DATA_DIR}")
            exit(1)
            
    else:
        # HC18 dataset configuration
        dataset_type = "hc18_dp"
        mask_suffix = "_mask.png"
        
        # Try augmented_processed first, fallback to training_set
        if os.path.exists(os.path.join(BASE_DATA_DIR, 'augmented_processed')):
            IMAGES_DIR = os.path.join(BASE_DATA_DIR, 'augmented_processed', 'images')
            MASKS_DIR = os.path.join(BASE_DATA_DIR, 'augmented_processed', 'masks')
        else:
            IMAGES_DIR = os.path.join(BASE_DATA_DIR, 'training_set')
            MASKS_DIR = os.path.join(BASE_DATA_DIR, 'training_set')
    
    print(f"Using dataset type: {dataset_type}")
    print(f"Base data directory: {BASE_DATA_DIR}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Masks directory: {MASKS_DIR}")
    print(f"Mask suffix: {mask_suffix}")
    
    # Output paths
    MODEL_SAVE_PATH = f'unet_{dataset_type}.pth'
    
    print(f"=== Step 1: Checking {dataset_type.title()} Data ===")
    if os.path.exists(IMAGES_DIR) and os.path.exists(MASKS_DIR):
        image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')]
        mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('_Annotation.png')]
        
        num_images = len(image_files)
        num_masks = len(mask_files)
        
        print(f"Found {num_images} images and {num_masks} masks in {dataset_type} dataset")
        
        if num_images == 0:
            print("Error: No images found in the dataset!")
            exit(1)
    else:
        print(f"{dataset_type.title()} data directories not found!")
        print(f"Images dir exists: {os.path.exists(IMAGES_DIR)}")
        print(f"Masks dir exists: {os.path.exists(MASKS_DIR)}")
        exit(1)
    
    # DP Configuration
    ENABLE_DP = False  # Set to False to train without differential privacy
    
    if ENABLE_DP:
        print("\n=== Step 2: Training HC18 + Multicenter U-Net with Differential Privacy ===")
        # More aggressive learning rate for DP
        dp_learning_rate = 1e-3  # Increased from 5e-4
        dp_batch_size = 16       # Increased from 4 for better DP performance
        dp_noise_multiplier = 0.8  # Reduced noise for faster convergence
        dp_max_grad_norm = 2.0     # Increased for less aggressive clipping
        print(f"Using DP-optimized settings:")
        print(f"  Learning rate: {dp_learning_rate}")
        print(f"  Batch size: {dp_batch_size}")
        print(f"  Noise multiplier: {dp_noise_multiplier}")
        print(f"  Max grad norm: {dp_max_grad_norm}")
    else:
        print("\n=== Step 2: Training HC18 + Multicenter U-Net without Differential Privacy ===")
        dp_learning_rate = 1e-4
        dp_batch_size = 4
        dp_noise_multiplier = 1.2
        dp_max_grad_norm = 1.0
        print(f"Using standard learning rate: {dp_learning_rate}")
    
    # Train the model
    model, history = train_model(
        IMAGES_DIR, 
        MASKS_DIR, 
        MODEL_SAVE_PATH, 
        epochs=30,
        batch_size=dp_batch_size, 
        learning_rate=dp_learning_rate,
        mask_suffix=mask_suffix,
        enable_dp=ENABLE_DP,
        noise_multiplier=dp_noise_multiplier,
        max_grad_norm=dp_max_grad_norm
    )

    print("\n=== Step 3: Plotting Training Metrics ===")
    plot_training_metrics(history, f'training_metrics_{dataset_type}.png')
    
    print("\n=== Step 4: Visualizing Sample Predictions ===")
    # Create a dataset for visualization (without transforms)
    vis_dataset = FetalHCDataset(
        IMAGES_DIR, 
        MASKS_DIR, 
        transform=None,
        target_size=(256, 256),
        mask_suffix=mask_suffix
    )
    
    # Use model._module for inference if DP is enabled, otherwise use model directly
    visualization_model = model._module if ENABLE_DP else model
    visualize_predictions(visualization_model, vis_dataset, device, save_path=f'sample_predictions_{dataset_type}.png')
    
    print(f"\n=== Training Complete! ===")
    print(f"Model saved as: {MODEL_SAVE_PATH}")
    print(f"Dataset: HC18 with multicenter U-Net architecture")
    print(f"Training samples: {num_images}")
    if ENABLE_DP:
        print("Differential Privacy: ENABLED")
        print(f"Noise Multiplier: 1.2")
        print(f"Max Grad Norm: 1.0")
    else:
        print("Differential Privacy: DISABLED")