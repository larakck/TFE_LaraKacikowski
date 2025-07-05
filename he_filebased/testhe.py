# he_filebased/test_full_pipeline.py

import numpy as np
import tenseal as ts
from he_filebased.utils.encryption_utils import (
    create_ckks_context,
    encrypt_model_parameters,
    decrypt_model_parameters,
)

def simulate_clients_encrypt(arrays, context):
    """
    Simule plusieurs clients qui encryptent chacun leur tableau.
    Retourne une liste de listes de bytes (ciphertexts par couche).
    """
    encrypted_clients = []
    for i, arr in enumerate(arrays):
        enc = encrypt_model_parameters([arr], context)
        print(f"[CLIENT {i}] clear      : {arr[:3]}")
        print(f"[CLIENT {i}] ciphertext size (bytes) : {len(enc[0])}")
        encrypted_clients.append(enc)
    return encrypted_clients

def server_aggregate_and_scale(all_encrypted, context, scale_factor=1000.0):
    """
    Simule le serveur qui :
      1) additionne homomorphiquement couche par couche
      2) multiplie par scale_factor
      3) renvoie la liste de bytes agrégés
    """
    n_clients = len(all_encrypted)
    n_layers = len(all_encrypted[0])
    agg_ciphertexts = []

    for layer_idx in range(n_layers):
        # addition HE
        agg_vec = ts.ckks_vector_from(context, all_encrypted[0][layer_idx])
        for client_vecs in all_encrypted[1:]:
            agg_vec += ts.ckks_vector_from(context, client_vecs[layer_idx])

        print(f"[SERVER] layer {layer_idx} sum decrypted  : {agg_vec.decrypt()[:3]}")
        # scaling
        agg_vec *= scale_factor
        print(f"[SERVER] layer {layer_idx} scaled decrypted: {agg_vec.decrypt()[:3]}")
        # serialize
        agg_ciphertexts.append(agg_vec.serialize())

    return agg_ciphertexts

def main():
    # 1) Création du contexte CKKS partagé
    ctx = create_ckks_context()

    # 2) Deux "clients" avec deux tableaux différents
    arr1 = np.array([1.0, 2.0, 3.0], dtype=float)
    arr2 = np.array([4.0, 5.0, 6.0], dtype=float)

    # 3) Chaque client encrypt
    all_encrypted = simulate_clients_encrypt([arr1, arr2], ctx)

    # 4) Serveur agrège + scale
    agg_ciphertexts = server_aggregate_and_scale(all_encrypted, ctx, scale_factor=1000.0)

    # 5) Serveur déchiffre en bout de pipeline pour vérif
    decrypted = decrypt_model_parameters(agg_ciphertexts, ctx)
    print(f"[FINAL] decrypted aggregated+scaled: {decrypted[0][:6]}")

if __name__ == "__main__":
    main()
