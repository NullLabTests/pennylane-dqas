"""
DQAS Time-Series Prediction Experiment.

Uses Differentiable Quantum Architecture Search to discover an
optimal variational circuit for time-series forecasting.

Usage:
    python -m experiments.timeseries [--n_qubits 4] [--n_layers 3]
"""

import argparse
import json
import os
import sys

import pennylane as qml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqas.core import DQASCircuit, OperationPool
from dqas.search import DQASSearcher
from dqas.utils import (
    generate_timeseries_data,
    print_architecture,
    count_params,
)
from experiments.config import DQASConfig


def main():
    parser = argparse.ArgumentParser(description="DQAS Time-Series Experiment")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--lr-circuit", type=float, default=0.05)
    parser.add_argument("--lr-arch", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    config = DQASConfig(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_epochs=args.n_epochs,
        lr_circuit=args.lr_circuit,
        lr_architecture=args.lr_arch,
        seed=args.seed,
        results_dir=args.results_dir,
    )

    os.makedirs(config.results_dir, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print("=" * 60)
    print("DQAS: Time-Series Prediction Experiment")
    print("=" * 60)
    print(f"Qubits: {config.n_qubits} | Layers: {config.n_layers}")
    print(f"Gate pool: {config.gate_pool}")
    print(f"Epochs: {config.n_epochs}")
    print()

    X, y = generate_timeseries_data(
        n_samples=config.n_train + config.n_val + config.n_test,
        seq_len=config.n_qubits,
        seed=config.seed,
    )

    X_train, y_train = X[:config.n_train], y[:config.n_train]
    X_val, y_val = X[config.n_train:config.n_train + config.n_val], y[config.n_train:config.n_train + config.n_val]
    X_test, y_test = X[config.n_train + config.n_val:], y[config.n_train + config.n_val:]

    pool = OperationPool(config.gate_pool)
    circuit = DQASCircuit(config.n_qubits, config.n_layers, op_pool=pool)

    dev = qml.device("default.qubit", wires=config.n_qubits)

    searcher = DQASSearcher(
        circuit=circuit,
        device=dev,
        lr_circuit=config.lr_circuit,
        lr_architecture=config.lr_architecture,
        temperature=config.temperature,
        temp_decay=config.temp_decay,
        min_temperature=config.min_temperature,
        arch_reg=config.arch_reg,
    )

    print("Starting architecture search...")
    history = searcher.search(
        X_train, y_train,
        X_val, y_val,
        n_epochs=config.n_epochs,
        inner_steps=config.inner_steps,
        verbose=args.verbose,
    )

    print("\nSearch complete. Evaluating discovered architecture...")
    results = searcher.evaluate(X_test, y_test)

    print_architecture(circuit)
    param_counts = count_params(circuit)

    summary = {
        "task": "timeseries",
        "config": {
            "n_qubits": config.n_qubits,
            "n_layers": config.n_layers,
            "gate_pool": config.gate_pool,
            "n_epochs": config.n_epochs,
            "lr_circuit": config.lr_circuit,
            "lr_architecture": config.lr_architecture,
        },
        "results": {
            "test_mse": results["test_loss"],
            "test_mse_rounded": round(results["test_loss"], 6),
        },
        "architecture": results["architecture"],
        "gate_names": results["gate_names"],
        "parameters": param_counts,
        "history": {k: v[-1] if isinstance(v, list) else v for k, v in history.items()},
    }

    out_path = os.path.join(config.results_dir, "timeseries_result.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved to {out_path}")
    print(f"Test MSE: {results['test_loss']:.6f}")
    print(f"Architecture parameters: {param_counts['architecture']}")
    print(f"Circuit parameters: {param_counts['circuit']}")
    print("Done.")


if __name__ == "__main__":
    main()
