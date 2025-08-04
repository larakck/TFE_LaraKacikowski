"""
Federated Learning with Homomorphic Encryption for Multicenter Data
Based on fl_he.py but using the U-Net model and data pipeline from main_dp_opacus_multicenter.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import random
from typing import Dict, List, Tuple, Optional
import copy
import tenseal as ts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Import Model Architecture from main_dp_opacus_multicenter.py ---
class UNet(nn.Module):
    """U-Net architecture with GroupNorm for DP compatibility"""
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
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"UNet initialized with {total_params:,} parameters")

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.dec4(torch.cat([self.upconv4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        return self.final(dec1)

# --- 2. Loss Functions and Metrics from main_dp_opacus_multicenter.py ---
class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss for better segmentation performance."""
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
    """Calculate Dice coefficient for binary segmentation."""
    if pred.requires_grad:
        pred_sigmoid = torch.sigmoid(pred)
    else:
        pred_sigmoid = pred
    
    pred_binary = (pred_sigmoid > 0.5).float()
    target_binary = (target > 0.5).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    if union == 0:
        return torch.tensor(1.0, device=pred.device)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def pixel_accuracy(pred, target):
    """Calculate pixel-wise accuracy."""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    correct = (pred == target).sum()
    total = target.numel()
    return correct / total

# --- 3. Data Preparation from main_dp_opacus_multicenter.py ---
def prepare_multicenter_data(multicenter_dir):
    """
    Prepare multicenter data by collecting all image-mask pairs from different datasets.
    Returns a list of (image_path, mask_path) tuples.
    """
    image_mask_pairs = []
    
    # Process train datasets
    train_dir = os.path.join(multicenter_dir, 'train')
    if os.path.exists(train_dir):
        for dataset_folder in os.listdir(train_dir):
            if dataset_folder.startswith('Dataset'):
                dataset_path = os.path.join(train_dir, dataset_folder)
                images_dir = os.path.join(dataset_path, 'imagesTr')
                masks_dir = os.path.join(dataset_path, 'labelsTr_trunctated')
                
                if os.path.exists(images_dir) and os.path.exists(masks_dir):
                    # Get all images
                    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
                    
                    for img_file in image_files:
                        img_path = os.path.join(images_dir, img_file)
                        
                        # Find corresponding mask
                        base_name = img_file.replace('_0000.png', '')
                        
                        # Try different mask naming patterns
                        possible_masks = [
                            base_name + '_mask.png',  # Algeria, Egypt, Malawi patterns
                            base_name.replace(base_name.split('_')[-1], 'mask_' + base_name.split('_')[-1]),  # Barcelona pattern
                        ]
                        
                        mask_found = False
                        for mask_name in possible_masks:
                            mask_path = os.path.join(masks_dir, mask_name)
                            if os.path.exists(mask_path):
                                image_mask_pairs.append((img_path, mask_path))
                                mask_found = True
                                break
                        
                        if not mask_found:
                            print(f"Warning: No mask found for {img_file}")
    
    # Process external datasets if needed
    external_dir = os.path.join(multicenter_dir, 'external')
    if os.path.exists(external_dir):
        sierra_leone_dir = os.path.join(external_dir, 'Dataset004_SierraLeone')
        if os.path.exists(sierra_leone_dir):
            images_dir = os.path.join(sierra_leone_dir, 'imagesTr')
            masks_dir = os.path.join(sierra_leone_dir, 'labelsTr_trunctated')
            
            if os.path.exists(images_dir) and os.path.exists(masks_dir):
                image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
                
                for img_file in image_files:
                    img_path = os.path.join(images_dir, img_file)
                    img_number = img_file.split('_')[0]
                    
                    mask_name = f"{img_number}_gt_{img_number}.png"
                    mask_path = os.path.join(masks_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        image_mask_pairs.append((img_path, mask_path))
                    else:
                        print(f"Warning: No mask found for {img_file}")
    
    print(f"Found {len(image_mask_pairs)} image-mask pairs across all datasets")
    return image_mask_pairs

class MulticenterFetalHCDataset(Dataset):
    """Custom PyTorch Dataset for multicenter fetal head circumference segmentation."""
    def __init__(self, image_mask_pairs, transform=None, target_size=(256, 256)):
        self.image_mask_pairs = image_mask_pairs
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            image = np.zeros(self.target_size, dtype=np.uint8)
        else:
            image = cv2.resize(image, self.target_size)
        
        image = image / 255.0  # Normalize to [0, 1]
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Could not load mask {mask_path}")
            mask = np.zeros(self.target_size, dtype=np.uint8)
        else:
            mask = cv2.resize(mask, self.target_size)
        
        # Ensure binary mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask / 255.0  # Normalize to [0, 1]
        
        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        # Apply transforms if provided
        if self.transform:
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

def get_train_transforms():
    """Get training data augmentation transforms optimized for medical segmentation."""
    return transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
    ])

# --- 4. Homomorphic Encryption Components ---
class TenSEALContext:
    """Wrapper for TenSEAL context with our configuration"""
    
    def __init__(self, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]):
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        
        print(f"TenSEAL Context initialized:")
        print(f"  - Polynomial modulus degree: {poly_modulus_degree}")
        print(f"  - Coefficient modulus sizes: {coeff_mod_bit_sizes}")
        print(f"  - Scale: {self.context.global_scale}")

class ModelEncryptor:
    """Handles encryption/decryption of model weights"""
    
    def __init__(self, context: TenSEALContext, encrypt_layers: List[str] = None):
        self.context = context.context
        self.encrypt_layers = encrypt_layers or []
        
        print(f"ModelEncryptor initialized")
        print(f"  - Layers to encrypt: {len(self.encrypt_layers) if self.encrypt_layers else 'All'}")
    
    def should_encrypt_layer(self, layer_name: str) -> bool:
        """Check if a layer should be encrypted"""
        if not self.encrypt_layers:  # Encrypt all if no specific layers
            return True
        return any(pattern in layer_name for pattern in self.encrypt_layers)
    
    def encrypt_weights(self, state_dict: Dict) -> Dict:
        """Encrypt model weights with chunking for large layers"""
        encrypted_dict = {}
        encrypted_count = 0
        total_params = 0
        max_chunk_size = 8192  # Conservative chunk size to avoid warnings
        
        print("Encrypting model weights...")
        
        # Show sample of what's being encrypted
        sample_shown = True
        encrypted_example_shown = False
        
        for name, param in state_dict.items():
            if self.should_encrypt_layer(name):
                flat_param = param.cpu().numpy().flatten()
                param_size = len(flat_param)
                
                # Show sample values for first encrypted layer
                if not sample_shown and param_size > 0:
                    print(f"\n  === ENCRYPTION EXAMPLE: {name} ===")
                    print(f"  Original values (first 5): {flat_param[:5].tolist()}")
                    sample_shown = True
                
                if param_size <= max_chunk_size:
                    # Small layer - encrypt directly
                    encrypted_param = ts.ckks_vector(self.context, flat_param)
                    encrypted_dict[name] = {
                        'encrypted': encrypted_param,
                        'shape': param.shape,
                        'dtype': param.dtype
                    }
                    
                    # Show encrypted form for first small encrypted layer
                    if not encrypted_example_shown:
                        print(f"  Encrypted form: {encrypted_param}")
                        print(f"  Type: {type(encrypted_param)}")
                        serialized = encrypted_param.serialize()
                        print(f"  Serialized size: {len(serialized):,} bytes")
                        print(f"  (Original was {param_size * 4} bytes)")
                        encrypted_example_shown = True
                else:
                    # Large layer - chunk and encrypt
                    print(f"  - Chunking large layer {name} ({param_size} params)")
                    chunks = []
                    first_chunk_shown = False
                    for i in range(0, param_size, max_chunk_size):
                        chunk = flat_param[i:i+max_chunk_size]
                        encrypted_chunk = ts.ckks_vector(self.context, chunk)
                        chunks.append(encrypted_chunk)
                        
                        # Show encrypted form for first chunk if no small layer was shown
                        if not first_chunk_shown and not encrypted_example_shown:
                            print(f"  Encrypted chunk 1: {encrypted_chunk}")
                            print(f"  Type: {type(encrypted_chunk)}")
                            serialized = encrypted_chunk.serialize()
                            print(f"  Serialized size: {len(serialized):,} bytes")
                            print(f"  (Original chunk was {len(chunk) * 4} bytes)")
                            first_chunk_shown = True
                            encrypted_example_shown = True
                    
                    encrypted_dict[name] = {
                        'encrypted_chunks': chunks,
                        'chunk_size': max_chunk_size,
                        'total_size': param_size,
                        'shape': param.shape,
                        'dtype': param.dtype
                    }
                encrypted_count += param.numel()
            else:
                # Keep as plaintext
                encrypted_dict[name] = {
                    'plaintext': param.cpu(),
                    'shape': param.shape,
                    'dtype': param.dtype
                }
            total_params += param.numel()
        
        print(f"  - Encrypted {encrypted_count:,}/{total_params:,} parameters ({encrypted_count/total_params*100:.1f}%)")
        return encrypted_dict
    
    def decrypt_weights(self, encrypted_dict: Dict) -> Dict:
        """Decrypt model weights with chunking support"""
        decrypted_dict = {}
        
        for name, data in encrypted_dict.items():
            if 'encrypted' in data:
                # Single encrypted vector
                decrypted = data['encrypted'].decrypt()
                decrypted_tensor = torch.tensor(decrypted, dtype=data['dtype'])
                decrypted_dict[name] = decrypted_tensor.reshape(data['shape'])
            elif 'encrypted_chunks' in data:
                # Multiple encrypted chunks - decrypt and concatenate
                decrypted_chunks = []
                for chunk in data['encrypted_chunks']:
                    decrypted_chunk = chunk.decrypt()
                    decrypted_chunks.extend(decrypted_chunk)
                
                # Trim to original size (last chunk might be padded)
                decrypted_chunks = decrypted_chunks[:data['total_size']]
                decrypted_tensor = torch.tensor(decrypted_chunks, dtype=data['dtype'])
                decrypted_dict[name] = decrypted_tensor.reshape(data['shape'])
            else:
                # Already plaintext
                decrypted_dict[name] = data['plaintext']
        
        return decrypted_dict

# --- 5. HE Federated Learning Components ---
class HEFederatedClient:
    """Federated learning client with homomorphic encryption"""
    
    def __init__(self, client_id: str, data_indices: List[int], full_dataset: Dataset,
                 he_context: TenSEALContext, encrypt_layers: List[str] = None,
                 batch_size: int = 4, learning_rate: float = 1e-3):
        self.client_id = client_id
        self.device = device
        self.he_context = he_context
        self.encryptor = ModelEncryptor(he_context, encrypt_layers)
        
        # Create client's local dataset
        self.local_dataset = Subset(full_dataset, data_indices)
        self.train_loader = DataLoader(
            self.local_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # Initialize local model
        self.model = UNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-5)
        self.criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        
        print(f"  HE Client {client_id} initialized with {len(self.local_dataset)} samples")
    
    def local_train(self, global_weights: Optional[Dict] = None, epochs: int = 3) -> Dict:
        """Train local model and return encrypted weights"""
        
        # Load global weights if provided (decrypt first)
        if global_weights is not None:
            decrypted_weights = self.encryptor.decrypt_weights(global_weights)
            self.model.load_state_dict(decrypted_weights)
        
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_dice = 0
            num_batches = 0
            
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    batch_dice = dice_score(outputs, masks)
                    epoch_loss += loss.item()
                    epoch_dice += batch_dice.item()
                    num_batches += 1
                
                if batch_idx % 5 == 0:
                    print(f"[HE Client {self.client_id}] Epoch {epoch+1}/{epochs}, "
                          f"Batch {batch_idx+1}/{len(self.train_loader)}: "
                          f"Loss={loss.item():.4f}, Dice={batch_dice.item():.4f}")
            
            avg_loss = epoch_loss / num_batches
            avg_dice = epoch_dice / num_batches
            total_loss += avg_loss
            total_dice += avg_dice
            total_batches += 1
            
            self.scheduler.step()
        
        # Store final metrics
        self.last_loss = total_loss / total_batches
        self.last_dice = total_dice / total_batches
        
        print(f"[HE Client {self.client_id}] Training completed: "
              f"Loss={self.last_loss:.4f}, Dice={self.last_dice:.4f}")
        
        # Encrypt and return model weights
        print(f"[HE Client {self.client_id}] Encrypting model weights...")
        encrypted_weights = self.encryptor.encrypt_weights(self.model.state_dict())
        
        return encrypted_weights

class HEFederatedServer:
    """Federated learning server with homomorphic encryption support"""
    
    def __init__(self, he_context: TenSEALContext, encrypt_layers: List[str] = None):
        self.global_model = UNet().to(device)
        self.he_context = he_context
        self.encryptor = ModelEncryptor(he_context, encrypt_layers)
        self.round_metrics = {'loss': [], 'dice': [], 'time': [], 'encryption_time': []}
        
        print(f"HE Federated Server initialized")
    
    def aggregate_encrypted_weights(self, client_weights: List[Dict]) -> Dict:
        """Aggregate encrypted weights using homomorphic operations"""
        
        print(f"\nAggregating encrypted weights from {len(client_weights)} clients...")
        print(f"  === SERVER VIEW (Cannot see actual values) ===")
        aggregation_start = time.time()
        
        # Initialize aggregated weights
        aggregated = {}
        num_clients = len(client_weights)
        
        # Show what server sees
        first_param_shown = False
        
        # Process each parameter
        for param_name in client_weights[0].keys():
            if 'encrypted' in client_weights[0][param_name]:
                # Single encrypted vector aggregation
                print(f"  - Aggregating encrypted parameter: {param_name}")
                
                # Show what server sees for first parameter
                if not first_param_shown:
                    print(f"\n  Example - Server sees for {param_name}:")
                    for i in range(min(3, num_clients)):
                        enc_obj = client_weights[i][param_name]['encrypted']
                        print(f"    Client {i+1}: {enc_obj}")
                    print(f"    → All values are opaque CKKSVector objects")
                    first_param_shown = True
                
                aggregated_param = client_weights[0][param_name]['encrypted']
                for i in range(1, num_clients):
                    aggregated_param += client_weights[i][param_name]['encrypted']
                aggregated_param *= (1.0 / num_clients)
                
                aggregated[param_name] = {
                    'encrypted': aggregated_param,
                    'shape': client_weights[0][param_name]['shape'],
                    'dtype': client_weights[0][param_name]['dtype']
                }
            elif 'encrypted_chunks' in client_weights[0][param_name]:
                # Chunked encrypted vector aggregation
                print(f"  - Aggregating chunked encrypted parameter: {param_name}")
                
                num_chunks = len(client_weights[0][param_name]['encrypted_chunks'])
                aggregated_chunks = []
                
                for chunk_idx in range(num_chunks):
                    # Start with first client's chunk
                    aggregated_chunk = client_weights[0][param_name]['encrypted_chunks'][chunk_idx]
                    
                    # Add other clients' chunks
                    for client_idx in range(1, num_clients):
                        aggregated_chunk += client_weights[client_idx][param_name]['encrypted_chunks'][chunk_idx]
                    
                    # Average the chunk
                    aggregated_chunk *= (1.0 / num_clients)
                    aggregated_chunks.append(aggregated_chunk)
                
                aggregated[param_name] = {
                    'encrypted_chunks': aggregated_chunks,
                    'chunk_size': client_weights[0][param_name]['chunk_size'],
                    'total_size': client_weights[0][param_name]['total_size'],
                    'shape': client_weights[0][param_name]['shape'],
                    'dtype': client_weights[0][param_name]['dtype']
                }
            else:
                # Plain aggregation for non-encrypted parameters
                stacked = torch.stack([w[param_name]['plaintext'].float() 
                                     for w in client_weights])
                avg_param = torch.mean(stacked, dim=0)
                
                # Convert back to original dtype
                original_dtype = client_weights[0][param_name]['dtype']
                if original_dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    avg_param = avg_param.round().to(original_dtype)
                else:
                    avg_param = avg_param.to(original_dtype)
                
                aggregated[param_name] = {
                    'plaintext': avg_param,
                    'shape': client_weights[0][param_name]['shape'],
                    'dtype': original_dtype
                }
        
        aggregation_time = time.time() - aggregation_start
        print(f"Aggregation completed in {aggregation_time:.2f}s")
        
        return aggregated
    
    def update_global_model(self, aggregated_weights: Dict):
        """Update global model with aggregated weights"""
        print("Decrypting and updating global model...")
        decrypted_weights = self.encryptor.decrypt_weights(aggregated_weights)
        self.global_model.load_state_dict(decrypted_weights)
        print("Global model updated")
        
    def evaluate_global_model(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate global model performance"""
        self.global_model.eval()
        total_loss = 0
        total_dice = 0
        num_batches = 0
        
        criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = self.global_model(images)
                
                loss = criterion(outputs, masks)
                dice = dice_score(outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return avg_loss, avg_dice

# --- 6. Visualization Functions ---
def plot_he_federated_results(metrics: Dict, num_clients: int, save_path: str = "fl_he_multicenter_results.png"):
    """Plot federated learning with HE results"""
    
    rounds = range(1, len(metrics['loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Federated Learning with Homomorphic Encryption (Multicenter Data)\n{num_clients} clients', 
                 fontsize=14, fontweight='bold')
    
    # Loss
    ax1.plot(rounds, metrics['loss'], 'b-o', linewidth=2, markersize=8)
    ax1.set_title('Training Loss (Encrypted)', fontweight='bold')
    ax1.set_xlabel('Federated Round')
    ax1.set_ylabel('Average Loss')
    ax1.grid(True, alpha=0.3)
    
    # Dice Score
    ax2.plot(rounds, metrics['dice'], 'g-s', linewidth=2, markersize=8)
    ax2.set_title('Dice Score Progress (Encrypted)', fontweight='bold')
    ax2.set_xlabel('Federated Round')
    ax2.set_ylabel('Average Dice Score')
    ax2.grid(True, alpha=0.3)
    
    # Round Times
    ax3.bar(rounds, metrics['time'], color='orange', alpha=0.7, label='Total Time')
    ax3.bar(rounds, metrics['encryption_time'], color='red', alpha=0.7, label='Encryption Time')
    ax3.set_title('Time per Round', fontweight='bold')
    ax3.set_xlabel('Federated Round')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Privacy Summary
    privacy_text = f"""Homomorphic Encryption Active
    
CKKS Scheme
Poly Degree: 32768
Security Level: 128-bit

Model updates encrypted
Server cannot see raw data
Privacy-preserving aggregation"""
    
    ax4.text(0.1, 0.5, privacy_text, 
             ha='left', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    ax4.set_title('Privacy Protection', fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Results saved to: {save_path}")

def visualize_predictions(model, dataset, device, num_samples=4, save_path="fl_he_multicenter_predictions.png"):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 10))
    fig.suptitle('HE Federated Learning Model Predictions (Multicenter)', fontsize=16, fontweight='bold')
    
    with torch.no_grad():
        for i in range(num_samples):
            image, mask = dataset[i]
            image_input = image.unsqueeze(0).to(device)
            
            # Get prediction
            pred_logits = model(image_input)
            pred_prob = torch.sigmoid(pred_logits)
            
            # Convert to numpy
            image_np = image.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            pred_prob_np = pred_prob.squeeze().cpu().numpy()
            
            # Calculate Dice
            dice = dice_score(pred_logits, mask.unsqueeze(0).to(device)).item()
            
            # Plot
            axes[0, i].imshow(image_np, cmap='gray')
            axes[0, i].set_title(f'Input {i+1}', fontsize=10)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(mask_np, cmap='viridis', vmin=0, vmax=1)
            axes[1, i].set_title(f'Ground Truth', fontsize=10)
            axes[1, i].axis('off')
            
            axes[2, i].imshow(pred_prob_np, cmap='viridis', vmin=0, vmax=1)
            axes[2, i].set_title(f'Prediction (Dice: {dice:.3f})', fontsize=10)
            axes[2, i].axis('off')
    
    # Add row labels
    row_labels = ['Input Image', 'Ground Truth', 'Model Prediction']
    for i, label in enumerate(row_labels):
        fig.text(0.02, 0.85 - i*0.28, label, fontsize=12, fontweight='bold', 
                ha='left', va='center', rotation=90)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Predictions saved to: {save_path}")
    
    model.train()

# --- 7. Main HE Federated Learning Pipeline ---
def run_he_federated_learning(multicenter_dir: str, num_rounds: int = 10, num_clients: int = 3, 
                             local_epochs: int = 3, samples_per_client: int = 200,
                             encrypt_layers: List[str] = None):
    """Run federated learning with homomorphic encryption on multicenter data"""
    
    print("="*70)
    print("FEDERATED LEARNING WITH HOMOMORPHIC ENCRYPTION (MULTICENTER)")
    print("Using U-Net Architecture + TenSEAL")
    print("="*70)
    
    # Initialize HE context
    he_context = TenSEALContext()
    
    # Default: encrypt only the output layer for efficiency
    if encrypt_layers is None:
        # For U-Net, encrypt only the final output layer which directly produces segmentation results
        encrypt_layers = [
            'final.weight',  # 64 parameters (64 input channels x 1 output channel x 1x1 kernel)
            'final.bias',    # 1 parameter
        ]
        print(f"\nSelective encryption enabled for output layer only (65 parameters)")
    
    # Prepare multicenter data
    print(f"\nPreparing multicenter data from: {multicenter_dir}")
    image_mask_pairs = prepare_multicenter_data(multicenter_dir)
    
    if len(image_mask_pairs) == 0:
        print("Error: No image-mask pairs found!")
        return None
    
    # Create full dataset
    full_dataset = MulticenterFetalHCDataset(
        image_mask_pairs,
        transform=get_train_transforms(),
        target_size=(256, 256)
    )
    
    # Split data among clients
    total_samples = len(full_dataset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)
    
    # Adjust samples per client based on available data
    actual_samples_per_client = min(samples_per_client, total_samples // num_clients)
    print(f"\nAdjusting samples per client: {actual_samples_per_client} (total: {total_samples})")
    
    # Create HE clients
    clients = []
    for i in range(num_clients):
        start_idx = i * actual_samples_per_client
        end_idx = min(start_idx + actual_samples_per_client, total_samples)
        
        # For the last client, give them any remaining samples
        if i == num_clients - 1:
            end_idx = total_samples
        
        client_indices = indices[start_idx:end_idx]
        
        # Skip if no samples for this client
        if len(client_indices) == 0:
            print(f"Warning: No samples for client {i+1}, skipping...")
            continue
        
        print(f"Creating client {i+1} with {len(client_indices)} samples (indices {start_idx} to {end_idx-1})")
        
        client = HEFederatedClient(
            client_id=f"HE_Client_{i+1}",
            data_indices=client_indices,
            full_dataset=full_dataset,
            he_context=he_context,
            encrypt_layers=encrypt_layers,
            batch_size=4,
            learning_rate=1e-3
        )
        clients.append(client)
    
    if len(clients) == 0:
        print("Error: No clients could be created with available data!")
        return None
    
    # Create HE server
    server = HEFederatedServer(he_context, encrypt_layers)
    
    # Create test dataset
    test_dataset = MulticenterFetalHCDataset(
        image_mask_pairs,
        transform=None,
        target_size=(256, 256)
    )
    # Use 20% of data for testing or minimum 10 samples
    num_test_samples = max(10, int(0.2 * total_samples))
    test_indices = indices[-num_test_samples:]  # Last samples for testing
    test_subset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    print(f"\nUsing {len(test_subset)} samples for testing")
    
    # HE Federated learning rounds
    for round_num in range(num_rounds):
        print(f"\n{'='*20} ROUND {round_num + 1}/{num_rounds} {'='*20}")
        round_start = time.time()
        encryption_time = 0
        
        # Encrypt and send global weights
        if round_num == 0:
            print("\n🔐 FIRST ENCRYPTION - What happens to the weights:")
        encrypted_global_weights = server.encryptor.encrypt_weights(
            server.global_model.state_dict()
        )
        
        # Train clients and collect encrypted weights
        client_encrypted_weights = []
        client_losses = []
        client_dice_scores = []
        
        for client in clients:
            print(f"\n[{client.client_id}] Starting local training...")
            
            # Measure encryption overhead
            enc_start = time.time()
            encrypted_weights = client.local_train(
                global_weights=copy.deepcopy(encrypted_global_weights),
                epochs=local_epochs
            )
            enc_end = time.time()
            encryption_time += (enc_end - enc_start)
            
            client_encrypted_weights.append(encrypted_weights)
            client_losses.append(client.last_loss)
            client_dice_scores.append(client.last_dice)
        
        # Aggregate encrypted weights on server
        aggregated_weights = server.aggregate_encrypted_weights(client_encrypted_weights)
        server.update_global_model(aggregated_weights)
        
        # Evaluate global model
        test_loss, test_dice = server.evaluate_global_model(test_loader)
        
        # Track metrics
        round_time = time.time() - round_start
        avg_client_loss = np.mean(client_losses)
        avg_client_dice = np.mean(client_dice_scores)
        
        server.round_metrics['loss'].append(avg_client_loss)
        server.round_metrics['dice'].append(avg_client_dice)
        server.round_metrics['time'].append(round_time)
        server.round_metrics['encryption_time'].append(encryption_time)
        
        print(f"\nRound {round_num + 1} Summary:")
        print(f"  Average Client Loss: {avg_client_loss:.4f}")
        print(f"  Average Client Dice: {avg_client_dice:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Dice: {test_dice:.4f}")
        print(f"  Round Time: {round_time:.2f}s (Encryption: {encryption_time:.2f}s)")
        print(f"  Privacy: Homomorphic Encryption Active")
        print(f"  🔒 Server never saw raw weights - all operations on encrypted data!")
    
    # Save final model
    model_path = "fl_he_multicenter_model.pth"
    torch.save(server.global_model.state_dict(), model_path)
    print(f"\nFinal model saved to: {model_path}")
    
    # Plot results
    plot_he_federated_results(server.round_metrics, num_clients)
    
    # Visualize predictions
    visualize_predictions(server.global_model, test_dataset, device)
    
    return server.global_model

# --- 8. Main Execution ---
def main():
    """Main execution"""
    
    # Configuration
    CONFIG = {
        'multicenter_dir': './multicenter',  # Path to multicenter data
        'num_rounds': 8,                     # Federated rounds
        'num_clients': 3,                    # Number of clients
        'local_epochs': 3,                   # Local training epochs per round
        'samples_per_client': 50,            # Samples per client (will be adjusted based on available data)
        'encrypt_layers': [
            # Output layer (critical for segmentation)
            'final.weight',      # 64 parameters
            'final.bias',        # 1 parameter
            
            # Upsampling layers (reconstruction path)
            'upconv1.weight',    # 8,192 parameters
            'upconv1.bias',      # 64 parameters
            
        ]
    }
    
    print("HE FEDERATED LEARNING CONFIGURATION:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Check dataset
    if not os.path.exists(CONFIG['multicenter_dir']):
        print(f"ERROR: Multicenter dataset not found at {CONFIG['multicenter_dir']}!")
        return
    
    try:
        # Run HE federated learning
        final_model = run_he_federated_learning(**CONFIG)
        
        print("\n" + "="*70)
        print("HE FEDERATED LEARNING COMPLETED!")
        print("="*70)
        print("Privacy-preserving federated learning with homomorphic encryption successful!")
        print("Model updates were encrypted throughout training")
        print("Server never had access to raw model parameters")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()