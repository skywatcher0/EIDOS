"""Generate a repeatable numeric regression table for the quick start."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "demo.csv")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, (300, 2))
    target = 1.5 + 2*x[:, 0] - 0.7*x[:, 1]**2 + rng.normal(0, 0.08, len(x))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "target": target}).to_csv(args.output, index=False, lineterminator="\n")
    print(str(args.output).encode('ascii', 'backslashreplace').decode())

if __name__ == "__main__":
    main()
