# he/utils/encryption_utils.py

import tenseal as ts
import numpy as np

def create_ckks_context(poly_mod_degree=16384, coeff_mod_bit_sizes=(60, 40, 40, 60)):
    # CKKS context public (client garde la clé privée)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40
    context.make_context_public()
    return context

def encrypt_model_parameters(parameters, context):
    """
    parameters: list of np.ndarray (model.parameters())
    returns: list of bytes (serialized CKKS vectors)
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
    returns: list of np.ndarray (decrypted tensors)
    """
    decrypted = []
    for data in encrypted_list:
        vec = ts.ckks_vector_from(context, data)
        vals = vec.decrypt()
        decrypted.append(np.array(vals))
    return decrypted
