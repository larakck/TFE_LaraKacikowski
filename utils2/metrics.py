import torch
import matplotlib.pyplot as plt

class BCEDiceLoss(torch.nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice_coeff = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice_coeff

    def forward(self, pred, target):
        return self.bce_weight * self.bce_loss(pred, target) + self.dice_weight * self.dice_loss(pred, target)

def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return torch.clamp(dice, 0.0, 1.0)

def pixel_accuracy(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = (target > 0.5).float()
    correct = (pred == target).sum()
    return torch.clamp(correct / target.numel(), 0.0, 1.0)

def iou_score(pred, target, smooth=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float().view(-1)
    target = (target > 0.5).float().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return torch.tensor(1.0 if intersection == 0 else 0.0)
    return torch.clamp((intersection + smooth) / (union + smooth), 0.0, 1.0)

def plot_training_metrics(history, save_path='training_metrics.png'):
    epochs = range(1, len(history['loss']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    titles = ['Loss', 'Accuracy', 'Dice Score', 'IoU Score']
    keys = ['loss', 'accuracy', 'dice', 'iou']
    colors = ['b', 'g', 'r', 'm']
    for ax, key, title, color in zip(axs.flat, keys, titles, colors):
        ax.plot(epochs, history[key], f'{color}-', label=f'Training {title}')
        ax.set_title(f'Training {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Training metrics plot saved to {save_path}")

def visualize_predictions(model, dataset, device, num_samples=4, save_path='sample_predictions.png'):
    import numpy as np
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
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
    print(f"Sample predictions saved to {save_path}")
    model.train()
