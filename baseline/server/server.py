import flwr as fl
import torch
import matplotlib.pyplot as plt
import os
from flwr.common import parameters_to_ndarrays

# Créer un dossier pour sauvegarder les poids et les graphiques
os.makedirs("weights", exist_ok=True)
os.makedirs("plots", exist_ok=True)

class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(self, noise_multiplier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.losses = []
        self.dices = []

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            weights = parameters_to_ndarrays(aggregated_parameters)
            weights_torch = [torch.tensor(w) for w in weights]
            torch.save(weights_torch, f"weights/nm_{self.noise_multiplier}_round_{server_round}.pth")

        if results:
            losses = [fit_res.metrics["loss"] for _, fit_res in results if fit_res.metrics and "loss" in fit_res.metrics]
            dices = [fit_res.metrics["dice"] for _, fit_res in results if fit_res.metrics and "dice" in fit_res.metrics]
            if losses:
                self.losses.append(sum(losses) / len(losses))
            if dices:
                self.dices.append(sum(dices) / len(dices))

        return aggregated_parameters, metrics_aggregated

# === Début du script principal ===
noise_multipliers = [0.0, 0.5, 1.0, 1.5]
all_losses = {}
all_dices = {}

for nm in noise_multipliers:
    print(f"\n==== Test pour noise_multiplier = {nm} ====")

    strategy = SaveMetricsStrategy(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        noise_multiplier=nm,
    )

    print("[SERVER] Starting FL server...")

    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=7),
        strategy=strategy,
    )

    all_losses[nm] = strategy.losses
    all_dices[nm] = strategy.dices

# === Tracer les graphiques ===

# Loss vs Rounds
plt.figure()
for nm, losses in all_losses.items():
    plt.plot(range(1, len(losses) + 1), losses, label=f"Noise {nm}")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Loss vs Rounds")
plt.legend()
plt.grid()
plt.savefig("plots/loss_vs_rounds.png")
plt.show()  # <= Affiche la figure

# Dice vs Rounds
plt.figure()
for nm, dices in all_dices.items():
    plt.plot(range(1, len(dices) + 1), dices, label=f"Noise {nm}")
plt.xlabel("Round")
plt.ylabel("Dice Score")
plt.title("Dice Score vs Rounds")
plt.legend()
plt.grid()
plt.savefig("plots/dice_vs_rounds.png")
plt.show()  # <= Affiche la figure

# Dice Final vs Noise Multiplier
final_dices = [all_dices[nm][-1] for nm in noise_multipliers]
plt.figure()
plt.plot(noise_multipliers, final_dices, marker='o')
plt.xlabel("Noise Multiplier")
plt.ylabel("Final Dice Score")
plt.title("Final Dice Score vs Noise Multiplier")
plt.grid()
plt.savefig("plots/final_dice_vs_noise.png")
plt.show()  # <= Affiche la figure

print("\n✅ Tous les tests terminés et graphiques enregistrés ET affichés ! 🚀")
