# test_inference_he_vs_plain.py
import torch
from utils.model import UNet
from utils.train_eval import evaluate, set_model_parameters
from utils.dataset import get_dataloaders
from he_amélioré.utils.encryption_utils import (
    create_ckks_context,
    encrypt_model_parameters,
    decrypt_model_parameters,
)

def main():
    # 1) Prépare data réelles (petit batch de validation)
    _, val_dl = get_dataloaders("multicenter/external/Dataset004_SierraLeone",
                                augment=False, batch_size=4)
    x, y = next(iter(val_dl))
    
    # 2) Poids initiaux (+ copy pour HE)
    net_plain     = UNet().eval()
    net_he_loaded = UNet().eval()
    original_weights = [p.detach().cpu().numpy() for p in net_plain.state_dict().values()]

    # 3) Chiffrement / déchiffrement
    ctx   = create_ckks_context()
    serial, counts = encrypt_model_parameters(original_weights, ctx)
    decrypted = decrypt_model_parameters(serial, ctx,
                                         shapes=[w.shape for w in original_weights],
                                         chunk_counts=counts)
    set_model_parameters(net_he_loaded, decrypted)

    # 4) Inférence des deux modèles
    with torch.no_grad():
        out_plain = net_plain(x)
        out_he    = net_he_loaded(x)

    # 5) Comparaison
    # Après : tolérance 1e-5
    if not torch.allclose(out_plain, out_he, atol=1e-5):
        diff = (out_plain - out_he).abs().max().item()
        print(f"✗ Outputs differ (max abs diff = {diff:.2e}) > tolérance 1e-5")
        exit(1)

    print("✔️ Inférence HE vs Plain identique pour un batch réel")

if __name__ == "__main__":
    main()
