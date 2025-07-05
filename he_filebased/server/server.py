# he_filebased/server/server.py
import os
import flwr as fl
import tenseal as ts
import numpy as np
import matplotlib.pyplot as plt

from he_filebased.utils.encryption_utils import create_ckks_context

# Création des dossiers nécessaires
os.makedirs("weights", exist_ok=True)
os.makedirs("plots", exist_ok=True)

class HEFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.he_context = create_ckks_context()
        self.losses = []
        self.dices = []
        print("[SERVER] HEFedAvg initialized")

    def aggregate_fit(self, rnd, results, failures):
        print(f"[SERVER] Round {rnd}: aggregation ({len(results)} client(s))")

        if not results:
            return None, {}

        # 🔹 Récupération des métriques
        round_losses = []
        round_dices = []
        for _, res in results:
            metrics = res.metrics
            if "loss" in metrics and "dice" in metrics:
                round_losses.append(metrics["loss"])
                round_dices.append(metrics["dice"])

        avg_loss = np.mean(round_losses) if round_losses else None
        avg_dice = np.mean(round_dices) if round_dices else None

        if avg_loss is not None:
            self.losses.append(avg_loss)
            print(f"[SERVER] Round {rnd} Avg Loss: {avg_loss:.4f}")
        if avg_dice is not None:
            self.dices.append(avg_dice)
            print(f"[SERVER] Round {rnd} Avg Dice: {avg_dice:.4f}")

        self.plot_metrics()

        # 🔹 Lecture fichiers chiffrés et agrégation
        all_chunks = []
        for _, res in results:
            fname = res.metrics.get("file")
            print(f"[SERVER] Round {rnd}: reading {fname}")
            with open(fname, "rb") as f:
                data = f.read()
            chunks = []
            i = 0
            while i < len(data):
                length = int.from_bytes(data[i : i + 4], "big")
                i += 4
                chunks.append(data[i : i + length])
                i += length
            all_chunks.append(chunks)

        # Agrégation homomorphe
        n = len(all_chunks)
        agg = []
        for layer_idx in range(len(all_chunks[0])):
            vec = ts.ckks_vector_from(self.he_context, all_chunks[0][layer_idx])
            for chunks in all_chunks[1:]:
                vec += ts.ckks_vector_from(self.he_context, chunks[layer_idx])
            vec = vec * (1.0 / n)
            agg.append(vec.serialize())

        # Sauvegarde de l’agrégation
        out = f"weights/encrypted_agg_round{rnd}.bin"
        with open(out, "wb") as f:
            for chunk in agg:
                f.write(len(chunk).to_bytes(4, "big"))
                f.write(chunk)
        print(f"[SERVER] Round {rnd}: aggregated -> {out}")

        return [], {}

    def plot_metrics(self):
        if not self.losses or not self.dices:
            return

        rounds = range(1, len(self.losses) + 1)
        plt.figure()
        plt.plot(rounds, self.losses, label="Loss")
        plt.plot(rounds, self.dices, label="Dice Score")
        plt.xlabel("Round")
        plt.ylabel("Value")
        plt.title("Federated Metrics per Round")
        plt.legend()
        plt.grid(True)
        plt.savefig("plots/metrics.png")
        plt.close()
        print("[SERVER] Plot saved to plots/metrics.png")

if __name__ == "__main__":
    print("[SERVER] Starting HE FL server...")
    strategy = HEFedAvg(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    )
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
