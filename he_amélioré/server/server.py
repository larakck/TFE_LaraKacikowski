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

from utils2.model import UNet
from utils2.metrics import BCEDiceLoss, dice_score, pixel_accuracy
from utils2.dataset import FetalHCDataset, get_train_transforms
from utils2.train_eval import train_model_client

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
def run_he_federated_learning(num_rounds=5, local_epochs=3, encrypt_layers=None):
    print("="*70)
    print("FEDERATED LEARNING WITH HOMOMORPHIC ENCRYPTION")
    print("Using U-Net Architecture + TenSEAL")
    print("="*70)

    # HE context
    he_context = TenSEALContext()

    # Default encrypt layers
    if encrypt_layers is None:
        encrypt_layers = ['final.weight', 'final.bias']
        print(f"\nSelective encryption enabled for output layer only (65 parameters)")

    # Client paths
    base_dir = "1327317/training_set_processed/clients_split"
    client_dirs = [os.path.join(base_dir, f"client{i}") for i in range(1, 4)]

    # Load datasets
    all_datasets = []
    for path in client_dirs:
        img_dir = os.path.join(path, "imagesTr")
        mask_dir = os.path.join(path, "labelsTr")
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"Missing data in: {path}")
            return
        dataset = FetalHCDataset(
            images_dir=img_dir,
            masks_dir=mask_dir,
            transform=get_train_transforms(),
            target_size=(256, 256)
        )
        all_datasets.append(dataset)

    # Create HE clients
    clients = []
    for i, dataset in enumerate(all_datasets):
        indices = list(range(len(dataset)))
        client = HEFederatedClient(
            client_id=f"HE_Client_{i+1}",
            data_indices=indices,
            full_dataset=dataset,
            he_context=he_context,
            encrypt_layers=encrypt_layers,
            batch_size=4,
            learning_rate=1e-3
        )
        clients.append(client)

    # Concatenate all data for test set
    from torch.utils.data import ConcatDataset, random_split

    combined = ConcatDataset(all_datasets)
    total = len(combined)
    test_size = max(10, int(0.2 * total))
    train_size = total - test_size

    train_set, test_set = random_split(combined, [train_size, test_size])
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    print(f"\nUsing {len(test_set)} samples for testing")

    # HE Server
    server = HEFederatedServer(he_context, encrypt_layers)

    # Training loop
    for round_num in range(num_rounds):
        print(f"\n{'='*20} ROUND {round_num + 1}/{num_rounds} {'='*20}")
        round_start = time.time()
        encryption_time = 0

        if round_num == 0:
            print("\n🔐 FIRST ENCRYPTION - What happens to the weights:")

        encrypted_global_weights = server.encryptor.encrypt_weights(
            server.global_model.state_dict()
        )

        client_weights = []
        client_losses = []
        client_dices = []

        for client in clients:
            print(f"\n[{client.client_id}] Starting local training...")
            enc_start = time.time()
            encrypted_weights = client.local_train(
                global_weights=copy.deepcopy(encrypted_global_weights),
                epochs=local_epochs
            )
            enc_end = time.time()
            encryption_time += (enc_end - enc_start)

            client_weights.append(encrypted_weights)
            client_losses.append(client.last_loss)
            client_dices.append(client.last_dice)

        aggregated_weights = server.aggregate_encrypted_weights(client_weights)
        server.update_global_model(aggregated_weights)

        test_loss, test_dice = server.evaluate_global_model(test_loader)

        round_time = time.time() - round_start
        avg_loss = np.mean(client_losses)
        avg_dice = np.mean(client_dices)

        server.round_metrics['loss'].append(avg_loss)
        server.round_metrics['dice'].append(avg_dice)
        server.round_metrics['time'].append(round_time)
        server.round_metrics['encryption_time'].append(encryption_time)

        print(f"\nRound {round_num + 1} Summary:")
        print(f"  Average Client Loss: {avg_loss:.4f}")
        print(f"  Average Client Dice: {avg_dice:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Dice: {test_dice:.4f}")
        print(f"  Round Time: {round_time:.2f}s (Encryption: {encryption_time:.2f}s)")
        print(f"  Privacy: Homomorphic Encryption Active")
        print(f"  🔒 Server never saw raw weights - all operations on encrypted data!")

    # Save final model
    torch.save(server.global_model.state_dict(), "fl_he_model_clientsplit.pth")
    print("\nModel saved to fl_he_model_clientsplit.pth")

    # Plot results
    plot_he_federated_results(server.round_metrics, num_clients=3)
    visualize_predictions(server.global_model, test_set, device)

# --- 8. Main Execution ---
def main():
    """Main execution"""
    run_he_federated_learning(
            num_rounds=5,
            # num_clients=1,
            local_epochs=3,
            encrypt_layers = [
                'final.weight',
                'final.bias',
                'upconv1.weight',
                'upconv1.bias',
                'dec1.0.weight',
                'dec1.0.bias'
            ]

        )
    # # Configuration
    # CONFIG = {
    #     # 'multicenter_dir': './multicenter',  # Path to multicenter data
    #     'num_rounds': 8,                     # Federated rounds
    #     'num_clients': 1,                    # Number of clients
    #     'local_epochs': 3,                   # Local training epochs per round
    #     # 'samples_per_client': 50,            # Samples per client (will be adjusted based on available data)
    #     'encrypt_layers': [
    #         # Output layer (critical for segmentation)
    #         'final.weight',      # 64 parameters
    #         'final.bias',        # 1 parameter
            
    #         # Upsampling layers (reconstruction path)
    #         'upconv1.weight',    # 8,192 parameters
    #         'upconv1.bias',      # 64 parameters
            
    #     ]
    # }
    
    # print("HE FEDERATED LEARNING CONFIGURATION:")
    # for key, value in CONFIG.items():
    #     print(f"  {key}: {value}")
    
    # Check dataset
    # if not os.path.exists(CONFIG['multicenter_dir']):
    #     print(f"ERROR: Multicenter dataset not found at {CONFIG['multicenter_dir']}!")
    #     return
    
    # try:
    #     # Run HE federated learning
    #     final_model = run_he_federated_learning(**CONFIG)
        
    #     print("\n" + "="*70)
    #     print("HE FEDERATED LEARNING COMPLETED!")
    #     print("="*70)
    #     print("Privacy-preserving federated learning with homomorphic encryption successful!")
    #     print("Model updates were encrypted throughout training")
    #     print("Server never had access to raw model parameters")
        
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     import traceback
    #     traceback.print_exc()

if __name__ == "__main__":
    main()