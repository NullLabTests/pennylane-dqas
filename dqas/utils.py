import torch
import numpy as np
from typing import List, Optional


def print_architecture(circuit) -> None:
    """Pretty-print the discovered circuit architecture."""
    info = circuit.summary()
    print(f"\n{'='*60}")
    print(f"DQAS Discovered Architecture")
    print(f"{'='*60}")
    print(f"  Qubits:        {info['n_qubits']}")
    print(f"  Layers:        {info['n_layers']}")
    print(f"  Parameters:    {info['n_params']}")
    print(f"\n  Layer breakdown:")
    for layer in info["architecture"]:
        sg = ", ".join(layer["single_qubit_gates"])
        tg = ", ".join(layer["two_qubit_gates"])
        print(f"    Layer {layer['layer']}:")
        print(f"      Single-qubit: [{sg}]")
        print(f"      Two-qubit:    [{tg}]")
    print(f"{'='*60}\n")


def generate_synthetic_data(
    n_samples: int = 200,
    n_features: int = 4,
    n_classes: int = 2,
    seed: int = 42,
    noise: float = 0.3,
):
    """Generate synthetic classification data for DQAS experiments."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = np.zeros(n_samples, dtype=int)

    centers = rng.randn(n_classes, n_features) * 2
    for i in range(n_samples):
        dists = [np.linalg.norm(X[i] - c) for c in centers]
        y[i] = np.argmin(dists)

    noise_mask = rng.rand(n_samples) < noise
    y[noise_mask] = rng.randint(0, n_classes, size=noise_mask.sum())

    X_t = torch.tensor(X, dtype=torch.float32)
    y_onehot = torch.zeros(n_samples, n_classes)
    y_onehot[range(n_samples), y] = 1.0

    return X_t, y_onehot


def generate_timeseries_data(
    n_samples: int = 200,
    seq_len: int = 4,
    seed: int = 42,
):
    """Generate synthetic time-series data."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, n_samples + seq_len)
    series = np.sin(t) + 0.3 * np.sin(3 * t) + 0.1 * rng.randn(n_samples + seq_len)

    X, y = [], []
    for i in range(n_samples):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])

    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    y_t = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
    return X_t, y_t


def count_params(circuit) -> int:
    """Count total trainable parameters."""
    arch_params = sum(p.numel() for l in circuit.layers for p in [l.arch_logits])
    circuit_params = circuit.circuit_params.numel()
    return {"architecture": arch_params, "circuit": circuit_params, "total": arch_params + circuit_params}
