"""Command-line training and prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="eidos", description="Train portable symbolic regression models and predict from CSV tables.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Select, fit, evaluate and save an EIDOS model")
    fit.add_argument("--input", type=Path, required=True, help="CSV or CSV.gz containing numeric features and target")
    fit.add_argument("--target", required=True)
    fit.add_argument("--features", nargs="+", help="Feature names; by default all columns other than target, environment and role")
    fit.add_argument("--environment", help="Column with environment/group labels")
    fit.add_argument("--role", help="Column with disc/diag/audit/cal/test labels")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--max-features", type=int, default=8)
    fit.add_argument("--alpha", type=float, default=0.1)
    fit.add_argument("--name", default="custom", help="Stable dataset identifier used for deterministic seeds")
    fit.add_argument("--output", type=Path, required=True, help="New output directory")
    predict = commands.add_parser("predict", help="Use a saved model on new rows")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import pandas as pd
    from ._io import write_json
    from .estimator import EIDOSRegressor
    try:
        table = pd.read_csv(args.input, float_precision="round_trip")
        if args.command == "fit":
            if args.output.exists() and any(args.output.iterdir()):
                parser.error("Output directory is not empty; choose a new --output path")
            excluded = {args.target, args.environment, args.role}
            features = args.features or [c for c in table if c not in excluded]
            if not features or excluded.intersection(features):
                parser.error("Features must exclude the target, environment and role columns")
            estimator = EIDOSRegressor(random_state=args.seed, max_features=args.max_features, alpha=args.alpha)
            estimator.fit(table[features], table[args.target],
                          environments=table[args.environment] if args.environment else None,
                          roles=table[args.role] if args.role else None, dataset_id=args.name)
            args.output.mkdir(parents=True, exist_ok=True)
            estimator.save(args.output / "model.json")
            write_json(args.output / "metrics.json", estimator.metrics_)
            pd.DataFrame({"row_id": range(len(table)), "role": estimator.role_assignments_}).to_csv(args.output / "roles.csv", index=False)
            estimator.test_predictions_.to_csv(args.output / "test_predictions.csv", index=False)
            write_json(args.output / "formula.json", {
                "expression": estimator.formula_, "expression_units": "discovery-standardized features and response",
                "preprocessing": estimator.record_["construction_context"],
                "note": "The displayed expression rounds constants. Use model.json for exact prediction.",
            })
            print(json.dumps({"output": str(args.output), "candidate": estimator.candidate_id_, "test_rmse": estimator.metrics_["test_rmse"]}))
        else:
            if args.output.resolve() in {args.input.resolve(), args.model.resolve()}:
                parser.error("Prediction output must differ from the input and model paths")
            if args.output.exists():
                parser.error("Prediction output already exists; choose a new --output path")
            estimator = EIDOSRegressor.load(args.model)
            prediction = estimator.predict(table[estimator.selected_features_])
            bounds = estimator.predict_interval(table[estimator.selected_features_])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"prediction": prediction, "lower": bounds[:, 0], "upper": bounds[:, 1]}).to_csv(args.output, index=False)
            print(json.dumps({"rows": len(prediction), "output": str(args.output)}))
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(2, f"eidos: {error}\n")
    return 0
