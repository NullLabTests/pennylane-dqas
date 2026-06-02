"""Shared configuration for DQAS experiments."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DQASConfig:
    n_qubits: int = 4
    n_layers: int = 4
    gate_pool: List[str] = field(default_factory=lambda: ["RX", "RY", "RZ", "H", "I", "CNOT", "CZ"])

    lr_circuit: float = 0.05
    lr_architecture: float = 0.01
    temperature: float = 1.0
    temp_decay: float = 0.98
    min_temperature: float = 0.1
    arch_reg: float = 0.001

    n_epochs: int = 30
    inner_steps: int = 3

    n_train: int = 100
    n_val: int = 50
    n_test: int = 50

    seed: int = 42
    results_dir: str = "results"

    def __post_init__(self):
        assert self.n_qubits >= 2, "Need at least 2 qubits"
        assert self.n_layers >= 1, "Need at least 1 layer"
