import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from utils2.model import UNet
from utils2.metrics import BCEDiceLoss, pixel_accuracy, dice_score, iou_score
from utils2.dataset import FetalHCDataset, get_train_transforms
def train_model(images_path, masks_path, model_save_path='unet_hc18_multicenter.pth',
                epochs=10, batch_size=4, learning_rate=1e-3, mask_suffix='_mask.png',
                enable_dp=True, noise_multiplier=1.0, max_grad_norm=1.0,
                val_split=0.15, test_split=0.15, shuffle_seed=42): 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Building multicenter U-Net model...")
    model = UNet()

    if not ModuleValidator.is_valid(model):
        print("Model contains incompatible layers. Attempting to fix...")
        model = ModuleValidator.fix(model)
    model = model.to(device)

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8 if enable_dp else 0.7,
        patience=8 if enable_dp else 5,
        min_lr=1e-5 if enable_dp else 5e-6,
        cooldown=3 if enable_dp else 2
    )
    print("Using DP-optimized learning rate scheduler" if enable_dp else "Using standard learning rate scheduler")

    # --- Dataset complet puis split train/val/test ---
    full_dataset = FetalHCDataset(
        images_path, masks_path,
        transform=get_train_transforms(),
        target_size=(256, 256),
        # mask_suffix=mask_suffix
    )
    n = len(full_dataset)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test
    g = torch.Generator().manual_seed(shuffle_seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val, n_test], generator=g)

    print(f"Split -> train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    # Loaders (DP seulement sur le train)
    privacy_engine = None
    target_delta = None

    if enable_dp:
        target_delta = 1 / n_train
        print(f"Setting DP target delta to {target_delta:.2e}")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
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
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        print("Training without differential privacy")

    # Val/Test loaders toujours “classiques”
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training on {len(train_ds)} samples | Batch size: {batch_size} | Epochs: {epochs}")

    model.train()
    training_history = {
        'loss': [], 'accuracy': [], 'dice': [],
        'val_loss': [], 'val_accuracy': [], 'val_dice': [],
        'test_loss': None, 'test_accuracy': None, 'test_dice': None
    }

    for epoch in range(epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        running_dice = 0.0
        num_batches = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            batch_accuracy = pixel_accuracy(outputs, masks)
            batch_dice = dice_score(outputs, masks)

            if torch.isnan(loss) or torch.isnan(batch_accuracy) or torch.isnan(batch_dice):
                print(f"Warning: NaN detected in batch {batch_idx}, skipping...")
                continue
            if torch.isinf(loss) or torch.isinf(batch_accuracy) or torch.isinf(batch_dice):
                print(f"Warning: Inf detected in batch {batch_idx}, skipping...")
                continue

            running_loss += loss.item()
            running_accuracy += batch_accuracy.item()
            running_dice += batch_dice.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(
                    f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                    f'Loss: {loss.item():.4f}, Acc: {batch_accuracy.item():.4f}, '
                    f'Dice: {batch_dice.item():.4f}'
                )

        if num_batches > 0:
            epoch_loss = running_loss / num_batches
            epoch_accuracy = running_accuracy / num_batches
            epoch_dice = running_dice / num_batches
        else:
            print(f"Warning: No valid batches in epoch {epoch+1}, using fallback values")
            epoch_loss, epoch_accuracy, epoch_dice = 1.0, 0.0, 0.0

        training_history['loss'].append(epoch_loss)
        training_history['accuracy'].append(epoch_accuracy)
        training_history['dice'].append(epoch_dice)

        # --- Évaluation validation à la fin de chaque epoch ---
        eval_model = model._module if enable_dp else model
        val_loss, val_dice, val_acc, _ = evaluate(eval_model, val_loader, criterion, device)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_acc)
        training_history['val_dice'].append(val_dice)

        # On garde ton choix de scheduler (sur la loss train)
        scheduler.step(val_loss)

        if enable_dp:
            epsilon = privacy_engine.get_epsilon(delta=target_delta)
            print(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Train Dice: {epoch_dice:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}, (ε = {epsilon:.2f}, δ = {target_delta:.2e})"
            )
        else:
            print(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Train Dice: {epoch_dice:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    # --- Évaluation test après entraînement ---
    print("\n=== Final Test Evaluation ===")
    eval_model = model._module if enable_dp else model
    test_loss, test_dice, test_acc, _ = evaluate(eval_model, test_loader, criterion, device)
    training_history['test_loss'] = test_loss
    training_history['test_accuracy'] = test_acc
    training_history['test_dice'] = test_dice
    print(f"[TEST] Loss: {test_loss:.4f} | Dice: {test_dice:.4f} | Acc: {test_acc:.4f}")

    print(f"\nTraining complete. Saving model to {model_save_path}")
    if enable_dp:
        torch.save(model._module.state_dict(), model_save_path)
    else:
        torch.save(model.state_dict(), model_save_path)

    print("\n=== Training Summary ===")
    print(f"Best Train Loss: {min(training_history['loss']):.4f}")
    print(f"Best Train Accuracy: {max(training_history['accuracy']):.4f}")
    print(f"Best Train Dice: {max(training_history['dice']):.4f}")
    print(f"Best Val Dice: {max(training_history['val_dice']):.4f}")
    print(f"Final Test Dice: {training_history['test_dice']:.4f}")

    return model, training_history

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_accuracy = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            dice = dice_score(outputs, masks)
            acc = pixel_accuracy(outputs, masks)

            if not torch.isnan(loss) and not torch.isnan(dice) and not torch.isnan(acc):
                total_loss += loss.item()
                total_dice += dice.item()
                total_accuracy += acc.item()
                total_batches += 1

    if total_batches == 0:
        return 1.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / total_batches
    avg_dice = total_dice / total_batches
    avg_acc = total_accuracy / total_batches

    return avg_loss, avg_dice, avg_acc, 0.0  # 0.0 = placeholder for dice_ellipse


def train_model_client(model, train_loader, optimizer, scheduler, criterion, epochs=1, device='cpu'):
    model.to(device)
    model.train()
    history = {'loss': [], 'accuracy': [], 'dice': []}

    for epoch in range(epochs):
        running_loss, running_acc, running_dice, running_iou = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            acc = pixel_accuracy(outputs, masks)
            dice = dice_score(outputs, masks)
            # iou = iou_score(outputs, masks)

            if not torch.isnan(loss) and not torch.isnan(dice) and not torch.isnan(acc) :
                running_loss += loss.item()
                running_acc += acc.item()
                running_dice += dice.item()
                # running_iou += iou.item()
                n_batches += 1

        if n_batches == 0:
            continue

        epoch_loss = running_loss / n_batches
        epoch_acc = running_acc / n_batches
        epoch_dice = running_dice / n_batches
        # epoch_iou = running_iou / n_batches

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['dice'].append(epoch_dice)
        # history['iou'].append(epoch_iou)

        scheduler.step(epoch_loss)

        print(f"[Client] Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f} | Acc: {epoch_acc:.4f}")

    return model, history
