import os
import torch
from utils2.dataset import FetalHCDataset, get_train_transforms
from utils2.model import UNet
from utils2.metrics import BCEDiceLoss, pixel_accuracy, dice_score, iou_score, plot_training_metrics, visualize_predictions
from utils2.train_eval import train_model


# --- Set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == '__main__':
    BASE_DATA_DIR = 'multicenter/external/Dataset004_SierraLeone'

    if 'multicenter' in BASE_DATA_DIR or 'Dataset' in BASE_DATA_DIR:
        dataset_type = "hc18_multicenter_dp"

        if os.path.exists(os.path.join(BASE_DATA_DIR, 'imagesTr')):
            IMAGES_DIR = os.path.join(BASE_DATA_DIR, 'imagesTr')
            if os.path.exists(os.path.join(BASE_DATA_DIR, 'labelsTr_trunctated')):
                MASKS_DIR = os.path.join(BASE_DATA_DIR, 'labelsTr_trunctated')
                mask_suffix = "_gt_"
                print("Using truncated masks for Sierra Leone dataset")
            else:
                MASKS_DIR = os.path.join(BASE_DATA_DIR, 'labelsTr')
                mask_suffix = "_gt_"
                print("Using filled masks for Sierra Leone dataset")
        elif os.path.exists(os.path.join(BASE_DATA_DIR, 'organized_training_data')):
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
        dataset_type = "hc18_dp"
        mask_suffix = "_mask.png"
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

    ENABLE_DP = False

    if ENABLE_DP:
        print("\n=== Step 2: Training HC18 + Multicenter U-Net with Differential Privacy ===")
        dp_learning_rate = 1e-3
        dp_batch_size = 16
        dp_noise_multiplier = 0.8
        dp_max_grad_norm = 2.0
    else:
        print("\n=== Step 2: Training HC18 + Multicenter U-Net without Differential Privacy ===")
        dp_learning_rate = 1e-4
        dp_batch_size = 4
        dp_noise_multiplier = 1.2
        dp_max_grad_norm = 1.0

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
    vis_dataset = FetalHCDataset(
        IMAGES_DIR,
        MASKS_DIR,
        transform=None,
        target_size=(256, 256),
        mask_suffix=mask_suffix
    )

    visualization_model = model._module if ENABLE_DP else model
    visualize_predictions(visualization_model, vis_dataset, device, save_path=f'sample_predictions_{dataset_type}.png')

    print(f"\n=== Training Complete! ===")
    print(f"Model saved as: {MODEL_SAVE_PATH}")
    print(f"Dataset: HC18 with multicenter U-Net architecture")
    print(f"Training samples: {num_images}")
    print("Differential Privacy: ENABLED" if ENABLE_DP else "Differential Privacy: DISABLED")
