import pandas as pd
import matplotlib.pyplot as plt

# 1) Read the CSV we just wrote
df = pd.read_csv("results.csv")

# 2) Plot Actual vs. Predicted side by side

# Plot: Actual labels
plt.figure(figsize=(6, 5))
plt.scatter(df["x1"], df["x2"], c=df["actual"], cmap="bwr", alpha=0.7)
plt.title("Actual Labels")
plt.xlabel("x1")
plt.ylabel("x2")
plt.colorbar(ticks=[0, 1], label="Class")
plt.tight_layout()
plt.show()

# Plot: Predicted labels
plt.figure(figsize=(6, 5))
plt.scatter(df["x1"], df["x2"], c=df["prediction"], cmap="bwr", alpha=0.7)
plt.title("Predicted Labels")
plt.xlabel("x1")
plt.ylabel("x2")
plt.colorbar(ticks=[0, 1], label="Class")
plt.tight_layout()
plt.show()

