import tenseal as ts
import numpy as np


def create_ckks_context(poly_mod_degree=16384, coeff_mod_bit_sizes=(60, 40, 40, 60)):
    """
    Crée un contexte CKKS contenant la clé secrète et les clés de Galois.
    Ne rend pas le contexte public : conserve la clé privée pour déchiffrement.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    # Génération de la clé secrète et des clés nécessaires
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def encrypt_model_parameters(parameters, context):
    """
    parameters: list of np.ndarray
    Retourne une liste de bytes (CKKS ciphertexts sérialisés)
    """
    encrypted = []
    for arr in parameters:
        flat = arr.flatten().tolist()
        vec = ts.ckks_vector(context, flat)
        encrypted.append(vec.serialize())
    return encrypted


def decrypt_model_parameters(encrypted_list, context):
    """
    encrypted_list: list of bytes (serialized CKKS vectors)
    Retourne une liste de np.ndarray déchiffrés
    """
    decrypted = []
    # Récupération explicite de la clé secrète du contexte
    secret_key = context.secret_key()

    for data in encrypted_list:
        # Reconstruction du vecteur CKKS avec la clé privée disponible
        vec = ts.ckks_vector_from(context, data)
        # Déchiffrement explicite avec la clé secrète
        vals = vec.decrypt(secret_key=secret_key)
        decrypted.append(np.array(vals))
    return decrypted


def aggregate_encrypted_parameters(all_compressed, context, compress_level=3):
    """
    all_compressed: list de listes de bytes, un par client
    Retourne la liste de bytes pour le modèle agrégé
    """
    n = len(all_compressed)
    # Décompression + reconstruction des CKKS vectors
    ckks_vectors = []
    for comp_list in all_compressed:
        layer_vecs = []
        for comp in comp_list:
            raw = ts.ckks_vector_from(context, comp)
            layer_vecs.append(raw)
        ckks_vectors.append(layer_vecs)

    # Agrégation homomorphe
    aggregated = []
    for layer_idx in range(len(ckks_vectors[0])):
        agg_vec = ckks_vectors[0][layer_idx]
        for vecs in ckks_vectors[1:]:
            agg_vec += vecs[layer_idx]
        agg_vec = agg_vec * (1.0 / n)
        aggregated.append(agg_vec.serialize())

    return aggregated
