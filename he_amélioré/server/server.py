# he_amélioré/server/server.py

import os
import gc
import psutil
import hashlib

import numpy as np
import matplotlib.pyplot as plt

import flwr as fl
import tenseal as ts

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from he_amélioré.utils.encryption_utils import create_ckks_context

# Répertoire des plots
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plots"))
os.makedirs(PLOTS_DIR, exist_ok=True)

def sha_of_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr).hexdigest()

class HEFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Contexte CKKS
        self.context = create_ckks_context()
        print(
            "[DEBUG CKKS] degree=", self.context.poly_modulus_degree,
            "mod_bits=", self.context.coeff_mod_bit_sizes,
            "scale=", self.context.global_scale,
        )

        # Historiques pour plots
        self.round_losses = []
        self.round_dices  = []

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        n_clients = len(results)
        print(f"[SERVER] Round {rnd}: aggregation with {n_clients} client(s)")

        # 1) Moyenne des métriques en clair
        losses = [res.metrics["loss"] for _, res in results]
        dices  = [res.metrics["dice"] for _, res in results]
        avg_loss = float(sum(losses) / n_clients)
        avg_dice = float(sum(dices)  / n_clients)
        self.round_losses.append(avg_loss)
        self.round_dices .append(avg_dice)
        print(f"[SERVER] ▶️ Mean loss={avg_loss:.4f}, mean dice={avg_dice:.4f}")

        # 2) Somme homomorphe brute
        encrypted_ndarrays_all = [
            parameters_to_ndarrays(res.parameters) for _, res in results
        ]
        n_layers = len(encrypted_ndarrays_all[0])
        aggregated = []
        for i in range(n_layers):
            ct = ts.ckks_vector_from(self.context, encrypted_ndarrays_all[0][i].tobytes())
            for client_chunks in encrypted_ndarrays_all[1:]:
                ct += ts.ckks_vector_from(self.context, client_chunks[i].tobytes())
            aggregated.append(np.frombuffer(ct.serialize(), dtype=np.uint8))

        # Debug hash layer0
        h0 = sha_of_array(aggregated[0])
        print(f"[SERVER DEBUG] 🔐 SHA256 layer0 = {h0}")

        gc.collect()
        print(f"[SERVER] RAM used: {psutil.virtual_memory().used / (1024**3):.2f} GB")

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
    print("[SERVER] Starting HE Flower server with plots…")
    strategy = HEFedAvg(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    )
    # On capture le History retourné
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )
  
    strategy.save_plots()
