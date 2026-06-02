import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split

from .core import DQASCircuit, OperationPool


class DQASSearcher:
    """
    Bi-level optimization for Differentiable Quantum Architecture Search.

    Implements the core algorithm from arXiv:2505.09653:
    - Inner loop: optimize circuit parameters θ on training data
    - Outer loop: optimize architecture parameters α on validation data
    - Gumbel-Softmax relaxation enables end-to-end differentiability
    """

    def __init__(
        self,
        circuit: DQASCircuit,
        device,
        lr_circuit: float = 0.05,
        lr_architecture: float = 0.01,
        temperature: float = 1.0,
        temp_decay: float = 0.99,
        min_temperature: float = 0.1,
        arch_reg: float = 0.001,
    ):
        self.circuit = circuit
        self.device = device
        self.lr_circuit = lr_circuit
        self.lr_architecture = lr_architecture
        self.temperature = temperature
        self.temp_decay = temp_decay
        self.min_temperature = min_temperature
        self.arch_reg = arch_reg

        self.arch_optimizer = torch.optim.Adam(
            [p for l in circuit.layers for p in [l.arch_logits]],
            lr=lr_architecture,
        )
        self.circuit_optimizer = torch.optim.Adam(
            [circuit.circuit_params],
            lr=lr_circuit,
        )

    def _architecture_entropy(self) -> torch.Tensor:
        """Compute entropy of architecture distribution as a regularizer."""
        entropy = 0.0
        for layer in self.circuit.layers:
            probs = F.softmax(layer.arch_logits, dim=-1)
            ent = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).mean()
            entropy = entropy + ent
        return entropy / len(self.circuit.layers)

    def _accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute classification accuracy."""
        return (preds.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()

    def search(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        n_epochs: int = 50,
        inner_steps: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the bi-level DQAS search.

        Args:
            X_train, y_train: training data (inner loop)
            X_val, y_val: validation data (outer loop)
            n_epochs: number of outer loop iterations
            inner_steps: circuit optimization steps per outer iteration

        Returns:
            history: training metrics
        """
        criterion = torch.nn.MSELoss()
        qnode = self.circuit.build_circuit_fn(self.device)
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "temperature": [],
            "entropy": [],
        }

        n_classes = y_train.shape[1] if y_train.ndim > 1 else 1
        n_qubits = self.circuit.n_qubits

        pbar = tqdm(range(n_epochs), desc="DQAS Search", disable=not verbose)
        for epoch in pbar:
            temp = max(self.temperature, self.min_temperature)
            self.temperature *= self.temp_decay

            arch_samples = self.circuit.sample_all(temperature=temp, hard=True)

            for _ in range(inner_steps):
                self.circuit_optimizer.zero_grad()
                preds_train = []
                for i in range(len(X_train)):
                    out = qnode(X_train[i], arch_samples)
                    preds_train.append(torch.stack(out))
                preds_train = torch.stack(preds_train)
                loss_train = criterion(preds_train, y_train)
                loss_train.backward()
                self.circuit_optimizer.step()

            self.arch_optimizer.zero_grad()
            preds_val = []
            for i in range(len(X_val)):
                out = qnode(X_val[i], arch_samples)
                preds_val.append(torch.stack(out))
            preds_val = torch.stack(preds_val)
            loss_val = criterion(preds_val, y_val)
            entropy = self._architecture_entropy()
            total_loss = loss_val + self.arch_reg * entropy
            total_loss.backward()
            self.arch_optimizer.step()

            with torch.no_grad():
                preds_train_all = []
                for i in range(len(X_train)):
                    out = qnode(X_train[i], arch_samples)
                    preds_train_all.append(torch.stack(out))
                preds_train_all = torch.stack(preds_train_all)
                train_loss = criterion(preds_train_all, y_train).item()

                preds_val_all = []
                for i in range(len(X_val)):
                    out = qnode(X_val[i], arch_samples)
                    preds_val_all.append(torch.stack(out))
                preds_val_all = torch.stack(preds_val_all)
                val_loss = criterion(preds_val_all, y_val).item()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["temperature"].append(temp)
            history["entropy"].append(entropy.item())

            if verbose and (epoch + 1) % 10 == 0:
                pbar.set_postfix({
                    "train": f"{train_loss:.4f}",
                    "val": f"{val_loss:.4f}",
                    "temp": f"{temp:.2f}",
                })

        return history

    def evaluate(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> Dict:
        """Evaluate the discovered architecture after search."""
        qnode = self.circuit.build_fixed_circuit_fn(self.device)
        criterion = torch.nn.MSELoss()

        with torch.no_grad():
            preds = []
            for i in range(len(X_test)):
                out = qnode(X_test[i])
                preds.append(torch.stack(out))
            preds = torch.stack(preds)
            loss = criterion(preds, y_test).item()

        arch = [l.derived_architecture().tolist() for l in self.circuit.layers]
        return {
            "test_loss": loss,
            "architecture": arch,
            "gate_names": self.circuit.pool.gate_names,
            "n_params": self.circuit.n_params,
        }

    def summary(self) -> Dict:
        """Print a summary of the discovered architecture."""
        arch = [l.derived_architecture() for l in self.circuit.layers]
        gate_names = self.circuit.pool.gate_names
        info = {
            "n_qubits": self.circuit.n_qubits,
            "n_layers": self.circuit.n_layers,
            "n_params": self.circuit.n_params,
            "architecture": [],
        }
        for li, layer_arch in enumerate(arch):
            choices = [gate_names[idx] for idx in layer_arch]
            info["architecture"].append({
                "layer": li,
                "single_qubit_gates": choices[:self.circuit.n_qubits],
                "two_qubit_gates": choices[self.circuit.n_qubits:],
            })
        return info
