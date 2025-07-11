# test_flower_local.py (version simplifiée)
from he_amélioré.client.client_3 import FlowerClient
from utils.train_eval import  evaluate
from utils.dataset import get_dataloaders
import torch

def main():
    client = FlowerClient(local_epochs=1)
    train_dl, val_dl = get_dataloaders("multicenter/external/Dataset004_SierraLeone",
                                       augment=True, batch_size=8)

    # 1) chiffrement initial par le client
    encrypted_params = client.get_parameters()

    # 2) le même client se déchiffre et charge les poids
    client.set_parameters(encrypted_params)

    # 3) évalue « avant »
    dice_before = evaluate(client.model, val_dl)
    print(f"Dice avant fit: {dice_before:.4f}")

    # 4) entraînement local (fit fait internement un set_parameters + train)
    encrypted_params, _, _ = client.fit(encrypted_params, config={})

    # 5) évalue « après »
    dice_after = evaluate(client.model, val_dl)
    print(f"Dice après fit: {dice_after:.4f}")

    assert dice_after > dice_before, "✗ Pas d'amélioration du Dice après fit()"
    print("✔️ Cycle FL+HE local OK, Dice s'améliore !")

if __name__ == "__main__":
    main()
