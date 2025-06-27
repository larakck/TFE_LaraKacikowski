import flwr as fl
import torch
import matplotlib.pyplot as plt
import os
from flwr.common import parameters_to_ndarrays

# Créer les dossiers si besoin
os.makedirs("weights", exist_ok=True)
os.makedirs("plots", exist_ok=True)

class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        noise_multiplier: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nm = noise_multiplier
        self.losses = []
        self.dices = []
        self.epsilons = []

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # Sauvegarder les poids
        if aggregated_parameters is not None:
            weights = parameters_to_ndarrays(aggregated_parameters)
            torch.save([torch.tensor(w) for w in weights],
                       f"weights/nm_{self.nm}_round_{server_round}.pth")

        # Enregistrer les métriques
        if results:
            self.losses.append(sum(r.metrics["loss"] for _, r in results) / len(results))
            self.dices.append(sum(r.metrics["dice"] for _, r in results) / len(results))
            self.epsilons.append(sum(r.metrics["epsilon"] for _, r in results) / len(results))

        return aggregated_parameters, metrics_aggregated


def main():
    # Liste des bruitages à tester
    noise_multipliers = [0.0, 0.5, 1.0, 1.5]

    all_losses = {}
    all_dices = {}
    all_epsilons = {}

    for nm in noise_multipliers:
        print(f"\n==== Federated Learning avec noise_multiplier = {nm} ====")

        # Cette fonction sera appelée AVANT CHAQUE fit()
        def send_config_fn(server_round: int):
            print(f"[Server] Sending noise_multiplier = {nm}")
            return {"noise_multiplier": nm}

        strategy = SaveMetricsStrategy(
            noise_multiplier=nm,
            fraction_fit=1.0,
            min_fit_clients=2,
            min_available_clients=2,
            on_fit_config_fn=send_config_fn,  # ← injection du noise
        )

        # Démarrer le serveur pour ce nm
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=5),
            strategy=strategy,
        )

        # Récupérer les résultats
        all_losses[nm] = strategy.losses
        all_dices[nm] = strategy.dices
        all_epsilons[nm] = strategy.epsilons

    # === Génération des graphiques ===
    plt.figure()
    for nm, losses in all_losses.items():
        plt.plot(range(1, len(losses)+1), losses, label=f"NM={nm}")
    plt.title("Loss vs Rounds")
    plt.xlabel("Round"); plt.ylabel("Loss")
    plt.legend(); plt.grid()
    plt.savefig("plots/loss_vs_rounds_all_nm.png")

    plt.figure()
    for nm, dices in all_dices.items():
        plt.plot(range(1, len(dices)+1), dices, label=f"NM={nm}")
    plt.title("Dice vs Rounds")
    plt.xlabel("Round"); plt.ylabel("Dice")
    plt.legend(); plt.grid()
    plt.savefig("plots/dice_vs_rounds_all_nm.png")

    plt.figure()
    for nm, eps in all_epsilons.items():
        plt.plot(range(1, len(eps)+1), eps, label=f"NM={nm}")
    plt.title("Epsilon vs Rounds")
    plt.xlabel("Round"); plt.ylabel("Epsilon")
    plt.legend(); plt.grid()
    plt.savefig("plots/epsilon_vs_rounds_all_nm.png")

    plt.figure()
    plt.plot(noise_multipliers, [d[-1] for d in all_dices.values()], marker='o')
    plt.title("Final Dice vs Noise Multiplier")
    plt.xlabel("Noise Multiplier"); plt.ylabel("Final Dice")
    plt.grid()
    plt.savefig("plots/final_dice_vs_noise.png")

    print("\n✅ Toutes les sessions ont été exécutées et les graphiques sont enregistrés.")

if __name__ == "__main__":
    main()
