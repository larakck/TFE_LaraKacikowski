import torch
import torch.nn as nn
from .metrics import dice_score, pixel_accuracy, visualize_predictions, ellipse_dice_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_dice = 0.0
    valid_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if x.size(0) == 0:
            print(f"[TRAIN] Skipping empty batch {batch_idx}")
            continue

        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ERROR] Batch {batch_idx} - Invalid loss: {loss.item()}")
            continue

        try:
            loss.backward()
        except RuntimeError as e:
            print(f"[ERROR] RuntimeError during backward pass at batch {batch_idx}: {e}")
            continue

        optimizer.step()

        probs = torch.sigmoid(logits)
        acc = pixel_accuracy(probs, y)
        dice = dice_score(probs, y)

        total_loss += loss.item()
        total_acc += acc
        total_dice += dice
        valid_batches += 1

        if batch_idx % 5 == 0:
            print(f"[TRAIN] Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f} | Dice: {dice:.4f} | Acc: {acc:.4f}")

    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
        avg_acc = total_acc / valid_batches
        avg_dice = total_dice / valid_batches
    else:
        avg_loss = avg_acc = avg_dice = 0.0

    print(f"[TRAIN] Epoch finished. Avg Loss: {avg_loss:.4f} | Avg Dice: {avg_dice:.4f} | Avg Acc: {avg_acc:.4f}")
    return avg_loss, avg_dice, avg_acc


def evaluate(model, dataloader, criterion=None):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_dice = 0.0
    valid_batches = 0
    total_dice_ellipse = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            if x.size(0) == 0:
                continue

            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)

            acc = pixel_accuracy(probs, y)
            dice_ell = ellipse_dice_score((probs > 0.5), y)
            dice = dice_score(probs, y)
            loss = criterion(logits, y).item() if criterion else 0.0

            total_loss += loss
            total_dice_ellipse += dice_ell
            total_acc += acc
            total_dice += dice
            valid_batches += 1

    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
        avg_acc = total_acc / valid_batches
        avg_dice = total_dice / valid_batches
        avg_dice_ellipse = total_dice_ellipse / valid_batches
    else:
        avg_loss = avg_dice_ellipse = avg_acc = avg_dice = 0.0

    print(f"[VAL] Avg Loss: {avg_loss:.4f} | Avg Dice: {avg_dice:.4f} | Ellipse Dice: {avg_dice_ellipse:.4f}| Avg Acc: {avg_acc:.4f}")
    return avg_loss, avg_dice, avg_acc, avg_dice_ellipse


def set_model_parameters(model: torch.nn.Module, parameters: list[torch.Tensor]) -> None:
    for param, new_param in zip(model.parameters(), parameters):
        param.data = new_param.data.clone()

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
