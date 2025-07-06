import tenseal as ts
import numpy as np
import torch

def create_ckks_context(
    poly_modulus_degree=4096,  # réduit (moitié de 8192)
    coeff_mod_bit_sizes=[30, 20, 30],  # moins de précision → moins de mémoire
    global_scale=2**10  # moins de bits → plus rapide
):
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = global_scale
    context.generate_galois_keys()
    return context

def encrypt_model_parameters(weights, context, chunk_size=2048):
    encrypted_chunks = []
    chunk_counts = []
    for w in weights:
        flat = w.flatten()
        count = 0
        for i in range(0, len(flat), chunk_size):
            chunk = flat[i : i + chunk_size]
            vec = ts.ckks_vector(context, chunk)
            encrypted_chunks.append(vec.serialize())  # ⚠️ assure-toi que c'est bien des bytes ici
            count += 1
        chunk_counts.append(count)
    return encrypted_chunks, chunk_counts


# Dans decrypt_model_parameters.py

def decrypt_model_parameters(parameters, context, shapes, chunk_counts):
    serialized_chunks = parameters  # ✅ car c’est déjà une liste de bytes
    decrypted_weights = []
    i = 0

    for shape, chunks in zip(shapes, chunk_counts):
        encrypted_parts = []
        for _ in range(chunks):
            chunk = serialized_chunks[i]
            if isinstance(chunk, np.ndarray):
                chunk = chunk.tobytes()
            vec = ts.CKKSVector.load(context, chunk)
            encrypted_parts.append(vec.decrypt())
            i += 1

        flat_weight = np.concatenate(encrypted_parts)
        decrypted_weight = torch.tensor(flat_weight.reshape(shape), dtype=torch.float32)
        decrypted_weights.append(decrypted_weight)

    return decrypted_weights
