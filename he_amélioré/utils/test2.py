# check_encryption_utils_context.py

import sys
import numpy as np
import tenseal as ts

# Import direct de votre fonction usine à contexte
from he_amélioré.utils.encryption_utils import create_ckks_context

def main():
    # 1) Créer le contexte CKKS avec vos paramètres par défaut
    ctx = create_ckks_context()

    # 2) Choisir une taille de vecteur < slots (ici 1024)
    vec_len = 1024
    print(f"✔️ Create context OK. Testing with a vector of length {vec_len}.")

    # 3) Générer un vecteur aléatoire
    orig = np.random.rand(vec_len).tolist()

    # 4) Chiffrement / déchiffrement
    enc = ts.ckks_vector(ctx, orig)
    dec = enc.decrypt()  # une liste Python de floats

    # 5) Mesurer l’erreur
    diff = np.abs(np.array(orig) - np.array(dec))
    max_err  = diff.max()
    mean_err = diff.mean()
    print(f"max error  = {max_err:.2e}")
    print(f"mean error = {mean_err:.2e}")

    # 6) Vérifier la tolérance
    tol = 1e-6
    if max_err < tol:
        print(f"✔️ Contexte CKKS valide (max_err < {tol})")
        sys.exit(0)
    else:
        print(f"✗ Échec : erreur {max_err:.2e} ≥ tolérance {tol}")
        sys.exit(1)

if __name__ == "__main__":
    main()
