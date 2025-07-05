# test.py

import tenseal as ts
import numpy as np
from he.utils.encryption_utils import create_ckks_context, encrypt_model_parameters, decrypt_model_parameters


def main():
    # 1) Crée un contexte CKKS avec la clé secrète
    ctx = create_ckks_context()

    # 2) Prépare un vecteur test
    original = np.array([1.23, 4.56, 7.89], dtype=float)
    print("Original :", original)

    # 3) Chiffrement
    encrypted_list = encrypt_model_parameters([original], ctx)
    print("Encrypted lengths:", [len(chunk) for chunk in encrypted_list])

    # 4) Déchiffrement
    decrypted_list = decrypt_model_parameters(encrypted_list, ctx)
    print("Decrypted :", decrypted_list[0])

    # 5) Vérification
    if np.allclose(original, decrypted_list[0], atol=1e-3):
        print("✅ Test réussi : déchiffrement OK")
    else:
        print("❌ Test échoué : vecteur différent")


if __name__ == "__main__":
    main()
