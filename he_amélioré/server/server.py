import flwr as fl
import tenseal as ts
import numpy as np
import psutil
import gc
import hashlib

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from he_fileless.utils.encryption_utils import create_ckks_context

def sha_of_array(arr):
    return hashlib.sha256(arr).hexdigest()

class HEFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context = create_ckks_context()

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        print(f"[SERVER] Round {rnd}: aggregation with {len(results)} client(s)")

        # 🔹 Extraction des poids chiffrés sous forme de ndarrays
        encrypted_ndarrays_all = [parameters_to_ndarrays(res.parameters) for _, res in results]
        n_clients = len(encrypted_ndarrays_all)
        n_layers = len(encrypted_ndarrays_all[0])
        aggregated = []

        for i in range(n_layers):
            # 🔁 Agrégation homomorphe
            vec = ts.ckks_vector_from(self.context, encrypted_ndarrays_all[0][i].tobytes())
            for client_chunks in encrypted_ndarrays_all[1:]:
                vec += ts.ckks_vector_from(self.context, client_chunks[i].tobytes())
            vec *= (1.0 / n_clients)
            serialized = vec.serialize()
            aggregated.append(np.frombuffer(serialized, dtype=np.uint8))

        # ✅ SHA-256 hash debug sur le layer 0 agrégé
        hash_val = sha_of_array(aggregated[0])
        print(f"[SERVER DEBUG] 🔐 SHA256 du poids agrégé (layer 0) : {hash_val}")

        gc.collect()
        print(f"[SERVER] RAM used: {psutil.virtual_memory().used / (1024**3):.2f} GB")

        return ndarrays_to_parameters(aggregated), {}

if __name__ == "__main__":
    print("[SERVER] Starting HE Flower server (safe config)...")
    strategy = HEFedAvg(fraction_fit=1.0, min_fit_clients=1, min_available_clients=1)
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
