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
        print(f"[SERVER] Agrégation du round {rnd}")

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

    def save_plots(self):
        """Trace et enregistre les courbes Loss/Dice après entraînement."""
        rounds = list(range(1, len(self.round_losses) + 1))

        # Loss
        plt.figure()
        plt.plot(rounds, self.round_losses, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Average Loss")
        plt.title("Loss per Round")
        plt.grid(True)
        path_loss = os.path.join(PLOTS_DIR, "loss_per_round.png")
        plt.savefig(path_loss)
        plt.close()
        print(f"[SERVER] Saved loss plot → {path_loss}")

        # Dice
        plt.figure()
        plt.plot(rounds, self.round_dices, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Average Dice")
        plt.title("Dice per Round")
        plt.grid(True)
        path_dice = os.path.join(PLOTS_DIR, "dice_per_round.png")
        plt.savefig(path_dice)
        plt.close()
        print(f"[SERVER] Saved dice plot → {path_dice}")


if __name__ == "__main__":
    print("[SERVER] Starting HE Flower server (no-avg config)…")
    strategy = HEFedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )
  
    strategy.save_plots()
