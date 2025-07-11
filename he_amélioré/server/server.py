import flwr as fl
import tenseal as ts
import numpy as np
import psutil
import gc
import hashlib

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from he_amélioré.utils.encryption_utils import create_ckks_context

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

        # 1) Récupère tous les chunks encryptés
        encrypted_ndarrays_all = [
            parameters_to_ndarrays(res.parameters) for _, res in results
        ]
        n_clients = len(encrypted_ndarrays_all)
        n_layers  = len(encrypted_ndarrays_all[0])
        aggregated = []

        # 2) Pour chaque layer, on fait uniquement la somme HE
        for i in range(n_layers):
            ct = ts.ckks_vector_from(
                self.context,
                encrypted_ndarrays_all[0][i].tobytes()
            )
            for client_chunks in encrypted_ndarrays_all[1:]:
                ct += ts.ckks_vector_from(
                    self.context,
                    client_chunks[i].tobytes()
                )
            # 🚫 Plus de division ni de rescale ici !
            aggregated.append(
                np.frombuffer(ct.serialize(), dtype=np.uint8)
            )

        # Debug hash
        hash_val = sha_of_array(aggregated[0])
        print(f"[SERVER DEBUG] 🔐 SHA256 layer0 = {hash_val}")

        gc.collect()
        print(f"[SERVER] RAM used: {psutil.virtual_memory().used / (1024**3):.2f} GB")

        # 3) On renvoie la somme chiffrée brute
        return ndarrays_to_parameters(aggregated), {}

if __name__ == "__main__":
    print("[SERVER] Starting HE Flower server (no-avg config)…")
    strategy = HEFedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )
