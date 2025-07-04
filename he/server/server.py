# he/server/server.py

import os
import sys
import flwr as fl
import tenseal as ts  # TenSEAL import
import torch
import numpy as np

from he.utils.encryption_utils import create_ckks_context

# Ajout du chemin racine
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

# Création du dossier pour stocker les fichiers chiffrés
os.makedirs("weights", exist_ok=True)

class HEFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Contexte CKKS partagé (clé privée non publique)
        self.he_context = create_ckks_context()
        print("[SERVER] HEFedAvg initialized")

    def aggregate_fit(self, rnd, results, failures):
        print(f"[SERVER] Round {rnd}: aggregation ({len(results)} client(s))")

        if not results:
            return None, {}

        # 1) Lire chaque fichier envoyé par les clients
        all_chunks = []
        for _, res in results:
            fname = res.metrics.get("file")
            print(f"[SERVER] Round {rnd}: reading {fname}")
            with open(fname, "rb") as f:
                data = f.read()
            # Parser le flux: 4 bytes longueur + ciphertext
            chunks = []
            i = 0
            while i < len(data):
                length = int.from_bytes(data[i : i + 4], "big")
                i += 4
                chunks.append(data[i : i + length])
                i += length
            all_chunks.append(chunks)

        # 2) Agrégation homomorphe: somme + moyenne
        n = len(all_chunks)
        agg = []
        for layer_idx in range(len(all_chunks[0])):
            # Reconstruire CKKSVector du premier client
            vec = ts.ckks_vector_from(self.he_context, all_chunks[0][layer_idx])
            # Ajouter les vecteurs des autres clients
            for chunks in all_chunks[1:]:
                vec += ts.ckks_vector_from(self.he_context, chunks[layer_idx])
            # Moyenne par multiplication par 1/n
            vec = vec * (1.0 / n)
            # Sérialiser le vecteur moyenné
            agg.append(vec.serialize())

        # 3) Écrire le blob agrégé dans un fichier
        out = f"weights/encrypted_agg_round{rnd}.bin"
        with open(out, "wb") as f:
            for chunk in agg:
                f.write(len(chunk).to_bytes(4, "big"))
                f.write(chunk)
        print(f"[SERVER] Round {rnd}: aggregated -> {out}")

        # 4) Retour léger: pas de paramètres lourds
        return [], {}

if __name__ == "__main__":
    print("[SERVER] Starting HE FL server (1 client expected)...")
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
