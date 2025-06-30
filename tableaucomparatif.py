import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Consolidated table of final Dice and Epsilon (approx from earlier context)
data = {
    "Noise Multiplier": [1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0],
    "Final Dice": [0.465, 0.438, 0.429, 0.445, 0.419, 0.435, 0.395],
    "Final Epsilon (approx)": [8.2, 6.3, 5.0, 4.2, 3.8, 3.2, 2.5],
    "Final Loss": [
        0.5063,  # NM=1.0
        0.5189,  # NM=1.5
        0.5281,  # NM=1.8
        0.5150,  # NM=2.0
        0.5257,  # NM=2.2
        0.5172,  # NM=2.5
        0.5455   # NM=3.0
    ]
}

df = pd.DataFrame(data)

# Display the table
import ace_tools as tools; tools.display_dataframe_to_user(name="Comparaison NM / Dice / Epsilon / Loss", dataframe=df)

# Optional plot for visual comparison
plt.figure(figsize=(10, 6))
sns.lineplot(x="Noise Multiplier", y="Final Dice", data=df, marker='o', label="Final Dice")
sns.lineplot(x="Noise Multiplier", y="Final Loss", data=df, marker='s', label="Final Loss")
sns.lineplot(x="Noise Multiplier", y="Final Epsilon (approx)", data=df, marker='^', label="Epsilon")
plt.title("Comparaison des performances selon le noise multiplier")
plt.xlabel("Noise Multiplier")
plt.ylabel("Valeurs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
