import os
import sys
import argparse
import copy
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# === Imports identiques à ton baseline ===
from utils2.model import UNet
from utils2.dataset import FetalHCDataset, get_train_transforms
from utils2.metrics import BCEDiceLoss, visualize_predictions
from utils2.train_eval import train_model_client, evaluate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("baseline/weights", exist_ok=True)
os.makedirs("baseline/plots", exist_ok=True)


class LocalClient:
    def __init__(self, data_dir: str, local_epochs: int = 1):
        self.local_epochs = int(local_epochs)
        self.model = UNet().to(DEVICE)
        self.criterion = BCEDiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        images_dir = os.path.join(data_dir, "imagesTr")
        masks_dir = os.path.join(data_dir, "labelsTr")

        full_dataset = FetalHCDataset(
            images_dir,
            masks_dir,
            transform=get_train_transforms(),
            target_size=(256, 256),  # suffix auto géré dans FetalHCDataset si tu as pris la version auto
        )

        # ---- Split 70/15/15 par client ----
        n = len(full_dataset)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val

        train_subset, val_subset, test_subset = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_dl = torch.utils.data.DataLoader(
            train_subset, batch_size=4, shuffle=True, num_workers=0, drop_last=False
        )
        self.val_dl = torch.utils.data.DataLoader(
            val_subset, batch_size=4, shuffle=False, num_workers=0
        )
        self.test_dl = torch.utils.data.DataLoader(
            test_subset, batch_size=4, shuffle=False, num_workers=0
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.7, patience=5, min_lr=5e-6, cooldown=2
        )

    def get_parameters(self):
        return [p.detach().cpu().clone() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new in zip(self.model.parameters(), parameters):
            p.data = new.to(DEVICE).detach().clone()

    def fit(self, parameters, config=None):
        print(f"[Client] Start training... (local epochs = {self.local_epochs})")
        if parameters is not None:
            self.set_parameters(parameters)

        _, history = train_model_client(
            self.model,
            self.train_dl,
            self.optimizer,
            self.scheduler,
            self.criterion,
            epochs=self.local_epochs,
            device=DEVICE,
        )

        val_loss, val_dice, val_acc, _ = evaluate(
            self.model, self.val_dl, self.criterion, DEVICE
        )

        print(f"[Client] Validation | Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | Acc: {val_acc:.4f}")

        return self.get_parameters(), len(self.train_dl.dataset), {
            "loss": float(history["loss"][-1]),
            "dice": float(val_dice),
            "accuracy": float(history["accuracy"][-1]),
        }

    def evaluate(self, parameters, config=None):
        if parameters is not None:
            self.set_parameters(parameters)
        val_loss, val_dice, val_acc, _ = evaluate(
            self.model, self.val_dl, self.criterion, DEVICE
        )
        print(f"[Client] Evaluation (VAL) | Dice: {val_dice:.4f} | Acc: {val_acc:.4f}")
        return float(val_loss), len(self.val_dl.dataset), {
            "dice": float(val_dice),
            "accuracy": float(val_acc),
        }

    # --- Évaluation sur le test set du client ---
    def evaluate_test(self, parameters):
        if parameters is not None:
            self.set_parameters(parameters)
        test_loss, test_dice, test_acc, _ = evaluate(
            self.model, self.test_dl, self.criterion, DEVICE
        )
        return float(test_loss), float(test_dice), float(test_acc)


def fedavg_weighted(param_lists, weights):
    n_clients = len(param_lists)
    assert n_clients == len(weights) and n_clients > 0

    agg = [torch.zeros_like(p, device="cpu") for p in param_lists[0]]
    total = float(sum(weights))

    for params, w in zip(param_lists, weights):
        coef = float(w) / total if total > 0 else 1.0 / n_clients
        for i, p in enumerate(params):
            agg[i] += p.detach().cpu() * coef

    return agg


def load_params_into_model(model: torch.nn.Module, params_list):
    with torch.no_grad():
        for p, new in zip(model.parameters(), params_list):
            p.copy_(new.to(next(model.parameters()).device))
    return model


def run_federated_training(client_dirs, num_rounds=20, local_epochs=[1]):
    # Uniformise: si un seul nombre fourni, applique à tous les clients
    if len(local_epochs) == 1:
        local_epochs = local_epochs * len(client_dirs)
    assert len(local_epochs) == len(client_dirs), "local_epochs doit avoir autant d'entrées que client_dirs"

    all_losses = []
    all_dices = []
    all_test_dices = []  # moyenne des TEST Dice clients par round

    # Crée les clients avec leurs epochs locales associées
    clients = [LocalClient(cd, local_epochs=local_epochs[i]) for i, cd in enumerate(client_dirs)]
    if len(clients) == 0:
        raise ValueError("Aucun client fourni.")

    # Paramètres initiaux globaux = paramètres du premier client
    global_params = clients[0].get_parameters()

    for rnd in range(1, num_rounds + 1):
        print(f"\n[ROUND {rnd}]")

        # Broadcast
        for c in clients:
            c.set_parameters(global_params)

        results_params = []
        results_sizes = []
        round_losses = []
        round_dices = []

        # Fit local
        for idx, c in enumerate(clients):
            print(f"[Server] configure_fit: sampled client {idx+1}/{len(clients)}")
            params, n_train, metrics = c.fit(parameters=global_params, config=None)
            results_params.append(params)
            results_sizes.append(n_train)
            if metrics and "loss" in metrics:
                round_losses.append(metrics["loss"])
            if metrics and "dice" in metrics:
                round_dices.append(metrics["dice"])

        # Aggregation FedAvg pondérée par #échantillons
        aggregated_params = fedavg_weighted(results_params, results_sizes)

        # Sauvegarde des poids agrégés du round
        weights_torch = [p.detach().cpu().clone() for p in aggregated_params]
        torch.save(weights_torch, f"baseline/weights/round_{rnd}.pth")

        # Deviens les nouveaux poids globaux
        global_params = aggregated_params

        # Moyennes train/val du round (basées sur métriques renvoyées par fit)
        if round_losses:
            all_losses.append(sum(round_losses) / len(round_losses))
        if round_dices:
            all_dices.append(sum(round_dices) / len(round_dices))

        # Évaluer le modèle global sur les TEST sets des clients
        test_dices_this_round = []
        for c in clients:
            test_loss, test_dice, test_acc = c.evaluate_test(parameters=global_params)
            test_dices_this_round.append(test_dice)
        if test_dices_this_round:
            avg_test_dice = sum(test_dices_this_round) / len(test_dices_this_round)
            all_test_dices.append(avg_test_dice)
            print(f"[Server] Round {rnd} | Avg TEST Dice across clients: {avg_test_dice:.4f}")

    # === Visualisation finale après le dernier round ===
    try:
        global_model = UNet().to(DEVICE)
        global_model = load_params_into_model(global_model, global_params)
        vis_dataset = clients[0].test_dl.dataset  # change l'index pour un autre client si besoin
        vis_path = f"baseline/plots/sample_predictions_round_{num_rounds}_multicenter.png"
        visualize_predictions(global_model, vis_dataset, DEVICE, save_path=vis_path)
        print(f"[Server] Visualisation enregistrée: {vis_path}")
    except Exception as e:
        print(f"[Server] Visualisation ignorée (erreur): {e}")

    # Courbe Loss (val moyenne par round)
    plt.figure()
    plt.plot(range(1, len(all_losses) + 1), all_losses, label="Val Loss (avg)")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Loss vs Rounds (multicenter)")
    plt.legend()
    plt.grid()
    plt.savefig("baseline/plots/loss_vs_rounds_multicenter.png")
    plt.close()

    # Courbe Dice (val moyenne par round)
    plt.figure()
    plt.plot(range(1, len(all_dices) + 1), all_dices, label="Val Dice (avg)")
    if all_test_dices:
        plt.plot(range(1, len(all_test_dices) + 1), all_test_dices, label="Test Dice (avg)", linestyle="--")
    plt.xlabel("Round")
    plt.ylabel("Dice Score")
    plt.title("Dice Score vs Rounds (multicenter)")
    plt.legend()
    plt.grid()
    plt.savefig("baseline/plots/dice_vs_rounds_multicenter.png")
    plt.close()

    print("\n✅ Entraînement terminé et graphiques enregistrés ! 🚀")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_dirs",
        type=str,
        nargs="+",
        default=[
            "multicenter/external/Dataset004_SierraLeone",
            "multicenter/train/Dataset002_Egypt",
            "multicenter/train/Dataset001_Algeria",
        ],
        help="Liste des dossiers clients (chacun contenant imagesTr/ et labelsTr/)",
    )
    parser.add_argument("--rounds", type=int, default=20, help="Nombre de rounds")
    parser.add_argument(
        "--local_epochs",
        type=int,
        nargs="+",
        default=[1],
        help="Epochs locales par client (ex: 2 1 1) ou un seul entier pour tous",
    )

    args = parser.parse_args()
    run_federated_training(args.client_dirs, num_rounds=args.rounds, local_epochs=args.local_epochs)
