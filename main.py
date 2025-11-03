from PIL._imaging import display
from matplotlib import pyplot as plt

from train_baselines import train_required_baselines_mlflow

out = train_required_baselines_mlflow()

# Required tables for your report:
print("Split:", out["split_note"])
display(out["Table1"])
display(out["Table2"])
