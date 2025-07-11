import tenseal as ts
import numpy as np
import torch



def create_ckks_context(
    poly_modulus_degree: int = 4096,
    coeff_mod_bit_sizes: list[int] = [30,20,30],
    global_scale: float = 2**23,
) -> ts.Context:
    """
    Contexte CKKS minimal avec keyswitching activé :
      - 4096 degree → 2048 slots
      - chaîne de moduli [60,20] = 80 bits (somme + 1 mul + rescale OK)
      - scale 2**20 → précision ≃1e‑6
      - génère relin_keys et galois_keys pour activer keyswitching
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    ctx.global_scale = global_scale

    # Clés nécessaires pour toute opération multiplicative :
    ctx.generate_relin_keys()    # relinearisation après une multiplication
    ctx.generate_galois_keys()   # rotations et autres key‑switching

    return ctx



def encrypt_model_parameters(weights, context, chunk_size=None):
    slot_count = context.poly_modulus_degree // 2  # = 2048
    if chunk_size is None:
        chunk_size = slot_count

    encrypted_chunks = []
    chunk_counts = []

    for w in weights:
        flat = w.flatten()
        n_chunks = 0
        for i in range(0, len(flat), chunk_size):
            block = flat[i : i + chunk_size].tolist()
            vec = ts.ckks_vector(context, block)
            encrypted_chunks.append(vec.serialize())
            n_chunks += 1
        chunk_counts.append(n_chunks)

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
