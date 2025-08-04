import torch
import matplotlib.pyplot as plt

def dice_score(pred, target, smooth=1e-6, threshold=0.5, verbose=False):
    """
    pred: Tensor [B, 1, H, W] — probabilités (sigmoidées)
    target: Tensor [B, 1, H, W] — binaires (0 ou 1)
    """
    if pred.ndim == 4:
        pred = pred.squeeze(1)
    if target.ndim == 4:
        target = target.squeeze(1)

    pred_bin = (pred > threshold).float()
    target = target.float()

    intersection = (pred_bin * target).sum(dim=(1, 2))
    union = pred_bin.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * intersection + smooth) / (union + smooth)

    if verbose:
        print(f"pred_bin sum: {pred_bin.sum().item()}, target sum: {target.sum().item()}, intersection: {intersection.item()}, dice: {dice.mean().item()}")

    return dice.mean()


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


def visualize_predictions(model, dataset, device, num_samples=4, save_path='sample_predictions.png'):
    """
    Visualize sample predictions from the model.
    """
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 12))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            image_input = image.unsqueeze(0).to(device)

            # Get prediction
            pred = model(image_input)

            # Convert to numpy with detach
            pred = pred.detach().cpu().squeeze().numpy()
            image_np = image.detach().cpu().squeeze().numpy()
            mask_np = mask.detach().cpu().squeeze().numpy()

            # Plot original image
            axes[0, i].imshow(image_np, cmap='gray')
            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')

            # Plot ground truth mask
            axes[1, i].imshow(mask_np, cmap='gray')
            axes[1, i].set_title(f'Ground Truth {i+1}')
            axes[1, i].axis('off')

            # Plot prediction
            axes[2, i].imshow(pred, cmap='gray')
            axes[2, i].set_title(f'Prediction {i+1}')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Sample predictions saved to {save_path}")
    model.train()
 

import torch
import numpy as np
import cv2

def ellipse_dice_score(pred_mask, target_mask, smooth=1e-6):
    """
    Calcule le Dice entre une ellipse ajustée sur la prédiction et le masque ground truth.
    pred_mask et target_mask : tensors (B, 1, H, W) avec valeurs 0/1.
    """
    dices = []

    for i in range(pred_mask.size(0)):  # batch loop
        pred = pred_mask[i, 0].cpu().numpy()
        target = target_mask[i, 0].cpu().numpy()
        
        # Convert pred to uint8 for OpenCV
        pred_bin = np.uint8(pred > 0.5) * 255
        contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ellipse_mask = np.zeros_like(pred_bin, dtype=np.uint8)

        if contours and len(contours[0]) >= 5:
            largest = max(contours, key=cv2.contourArea)
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)

        ellipse_mask = torch.tensor(ellipse_mask / 255.0, dtype=torch.float32).to(pred_mask.device)

        # Compute Dice
        intersection = (ellipse_mask * torch.tensor(target, device=pred_mask.device)).sum()
        union = ellipse_mask.sum() + torch.tensor(target, device=pred_mask.device).sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dices.append(dice.item())

    return sum(dices) / len(dices) if dices else 0.0
