"""Run after installing EIDOS; paths are independent of the current directory."""
from pathlib import Path
import pandas as pd
from eidos import EIDOSRegressor

root = Path(__file__).resolve().parent
data = pd.read_csv(root / "demo.csv")
model = EIDOSRegressor(random_state=42).fit(data[["x", "z"]], data["target"])
output = root.parent / "results" / "python_api"
output.mkdir(parents=True, exist_ok=True)
model.save(output / "model.json")
loaded = EIDOSRegressor.load(output / "model.json")
print(model.metrics_)
print(loaded.predict(data[["x", "z"]].head()))
print(loaded.predict_interval(data[["x", "z"]].head()))
