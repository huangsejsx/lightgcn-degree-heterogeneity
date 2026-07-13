import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
results_dir = Path("results")
df = pd.read_csv(results_dir / "edge_structure_by_item_group.csv")
df["curvature_magnitude"] = df["forman_ricci"].abs()
order = ["low", "medium", "high"]
df["item_group"] = pd.Categorical(df["item_group"], categories=order, ordered=True)
df = df.sort_values("item_group")
plt.figure(figsize=(6, 4))
plt.bar(df["item_group"].astype(str), df["curvature_magnitude"])
plt.xlabel("Item degree group")
plt.ylabel("Mean curvature magnitude")
plt.title("Forman-Ricci Curvature Magnitude by Item Group")
plt.tight_layout()
plt.savefig(results_dir / "forman_ricci_magnitude_by_item_group.png", dpi=300)
plt.close()
print("Saved to results/forman_ricci_magnitude_by_item_group.png")