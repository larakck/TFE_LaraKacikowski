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
from utils.model import UNet
from utils.dataset import ClientDataset

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
            cooldown=3       # Longer cooldown for D
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
    train_dataset = ClientDataset(images_path, split="train", img_size=256, augment=True)
    
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
    # Update these paths to match your actual dataset location
    BASE_DATA_DIR = './1327317/training_set_processed'
    dataset_type = "hc18_multicenter_dp"
    MODEL_SAVE_PATH = f'unet_{dataset_type}.pth'

    print(f"=== Step 1: Checking {dataset_type.title()} Data ===")
    images_dir = os.path.join(BASE_DATA_DIR, 'imagesTr')
    masks_dir = os.path.join(BASE_DATA_DIR, 'labelsTr')
    if os.path.exists(images_dir) and os.path.exists(masks_dir):
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
        print(f"Found {len(image_files)} images and {len(mask_files)} masks in {dataset_type} dataset")
        if len(image_files) == 0:
            print("Error: No images found in the dataset!")
            exit(1)
    else:
        print("Image or mask directory not found!")
        exit(1)

    # DP Configuration
    ENABLE_DP = True  # Set to False to train without differential privacy
    
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
        BASE_DATA_DIR,
        BASE_DATA_DIR,
        MODEL_SAVE_PATH, 
        epochs=30,
        batch_size=dp_batch_size,  
        learning_rate=dp_learning_rate,
        enable_dp=ENABLE_DP,
        noise_multiplier=dp_noise_multiplier,
        max_grad_norm=dp_max_grad_norm
    )

    print("\n=== Step 3: Plotting Training Metrics ===")
    plot_training_metrics(history, f'training_metrics_{dataset_type}.png')
    
    print("\n=== Step 4: Visualizing Sample Predictions ===")
    # Create a dataset for visualization (without transforms)
    vis_dataset = ClientDataset(BASE_DATA_DIR, split="val", img_size=256, augment=False)

    # Use model._module for inference if DP is enabled, otherwise use model directly
    visualization_model = model._module if ENABLE_DP else model
    visualize_predictions(visualization_model, vis_dataset, device, save_path=f'sample_predictions_{dataset_type}.png')
    
    print(f"\n=== Training Complete! ===")
    print(f"Model saved as: {MODEL_SAVE_PATH}")
    print(f"Dataset: HC18 with multicenter U-Net architecture")
    print(f"Training samples: {len(image_files)}")

    if ENABLE_DP:
        print("Differential Privacy: ENABLED")
        print(f"Noise Multiplier: 1.2")
        print(f"Max Grad Norm: 1.0")
    else:
        print("Differential Privacy: DISABLED")