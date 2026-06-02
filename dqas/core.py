import pennylane as qml
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable


class OperationPool:
    """
    Registry of quantum operations available for architecture search.
    Each operation has a name, arity (1 or 2 qubits), and parameter count.
    """

    GATE_REGISTRY = {
        "RX": {"arity": 1, "params": 1, "fn": lambda p, w: qml.RX(*p, wires=w)},
        "RY": {"arity": 1, "params": 1, "fn": lambda p, w: qml.RY(*p, wires=w)},
        "RZ": {"arity": 1, "params": 1, "fn": lambda p, w: qml.RZ(*p, wires=w)},
        "H": {"arity": 1, "params": 0, "fn": lambda p, w: qml.Hadamard(wires=w)},
        "I": {"arity": 1, "params": 0, "fn": lambda p, w: qml.Identity(wires=w)},
        "CNOT": {"arity": 2, "params": 0, "fn": lambda p, w: qml.CNOT(wires=w)},
        "CZ": {"arity": 2, "params": 0, "fn": lambda p, w: qml.CZ(wires=w)},
        "SWAP": {"arity": 2, "params": 0, "fn": lambda p, w: qml.SWAP(wires=w)},
    }

    def __init__(self, gate_names: Optional[List[str]] = None):
        self.gates = {}
        names = gate_names or list(self.GATE_REGISTRY.keys())
        for name in names:
            if name not in self.GATE_REGISTRY:
                raise ValueError(f"Unknown gate: {name}. Available: {list(self.GATE_REGISTRY.keys())}")
            self.gates[name] = self.GATE_REGISTRY[name]

    @property
    def num_ops(self) -> int:
        return len(self.gates)

    @property
    def gate_names(self) -> List[str]:
        return list(self.gates.keys())

    @property
    def max_params(self) -> int:
        return max(g["params"] for g in self.gates.values())

    def apply(self, gate_name: str, params: torch.Tensor, wires) -> None:
        spec = self.gates[gate_name]
        p = params[:spec["params"]] if spec["params"] > 0 else []
        if len(wires) == 1:
            spec["fn"](p, wires[0])
        else:
            spec["fn"](p, wires)

    def requires_params(self, gate_name: str) -> bool:
        return self.gates[gate_name]["params"] > 0

    def arity(self, gate_name: str) -> int:
        return self.gates[gate_name]["arity"]


class MixedLayer:
    """
    A circuit layer with differentiable architecture search via Gumbel-Softmax.

    At each qubit position, the layer maintains architecture logits over
    the operation pool. During search, Gumbel-Softmax samples a discrete
    operation differentiably. After search, argmax yields the final architecture.

    Two-qubit gates are applied between adjacent qubits in a chain topology,
    with the chosen operation acting on pairs (0,1), (1,2), ..., (n-2, n-1).
    """

    def __init__(self, n_qubits: int, op_pool: OperationPool, layer_id: int = 0):
        self.n_qubits = n_qubits
        self.pool = op_pool
        self.layer_id = layer_id
        n_single = n_qubits
        n_two = n_qubits - 1
        self.n_positions = n_single + n_two
        self.arch_logits = torch.nn.Parameter(
            torch.zeros(self.n_positions, self.pool.num_ops)
        )

    def sample(self, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        """Sample a discrete architecture via Gumbel-Softmax.

        Returns one-hot tensor [n_positions, num_ops] with straight-through
        gradient estimator when hard=True.
        """
        if hard:
            hard_sample = F.gumbel_softmax(self.arch_logits, tau=temperature, hard=True)
            soft_sample = F.gumbel_softmax(self.arch_logits, tau=temperature, hard=False)
            return hard_sample.detach() + soft_sample - soft_sample.detach()
        return F.gumbel_softmax(self.arch_logits, tau=temperature, hard=False)

    def derived_architecture(self) -> torch.Tensor:
        """Return the final discrete architecture (integer indices)."""
        return self.arch_logits.argmax(dim=-1)

    def build(
        self,
        arch_sample: torch.Tensor,
        params: torch.Tensor,
        param_offset: int,
    ) -> Tuple[List, int]:
        """Build quantum operations from a sampled architecture.

        Args:
            arch_sample: one-hot [n_positions, num_ops] from sample()
            params: flat tensor of all circuit parameters
            param_offset: starting index into params for this layer

        Returns:
            ops: list of (gate_name, wires, param_slice) tuples
            param_offset + consumed: updated offset
        """
        ops = []
        consumed = 0
        indices = arch_sample.argmax(dim=-1)

        pos = 0
        for q in range(self.n_qubits):
            op_name = self.pool.gate_names[indices[pos]]
            n_p = self.pool.gates[op_name]["params"]
            p = params[param_offset + consumed: param_offset + consumed + n_p]
            ops.append((op_name, [q], p, n_p))
            consumed += n_p
            pos += 1

        for q in range(self.n_qubits - 1):
            op_name = self.pool.gate_names[indices[pos]]
            n_p = self.pool.gates[op_name]["params"]
            p = params[param_offset + consumed: param_offset + consumed + n_p]
            ops.append((op_name, [q, q + 1], p, n_p))
            consumed += n_p
            pos += 1

        return ops, param_offset + consumed


class DQASCircuit:
    """
    A full variational quantum circuit with differentiable architecture search.

    Composed of multiple MixedLayer instances stacked sequentially. The circuit
    can be configured with a searchable architecture or a fixed derived architecture.

    Architecture search operates over:
      - Single-qubit gate choice at each qubit per layer (RX, RY, RZ, H, I)
      - Two-qubit gate choice between adjacent qubits per layer (CNOT, CZ, SWAP, I)
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        op_pool: Optional[OperationPool] = None,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.pool = op_pool or OperationPool()
        self.params_per_position = self.pool.max_params
        n_params_per_layer = n_qubits * self.params_per_position + (n_qubits - 1) * self.params_per_position
        self.n_params = n_layers * n_params_per_layer
        self.circuit_params = torch.nn.Parameter(0.01 * torch.randn(self.n_params))
        self.layers = [
            MixedLayer(n_qubits, self.pool, layer_id=i)
            for i in range(n_layers)
        ]

    @property
    def arch_logits(self):
        return [l.arch_logits for l in self.layers]

    def sample_all(self, temperature: float = 1.0, hard: bool = False) -> List[torch.Tensor]:
        """Sample architectures for all layers."""
        return [l.sample(temperature, hard) for l in self.layers]

    def derive_all(self) -> List[torch.Tensor]:
        """Derive final architectures for all layers."""
        return [l.derived_architecture() for l in self.layers]

    def build_circuit_fn(
        self,
        device: qml.Device,
    ) -> Callable:
        """Create a PennyLane QNode from the current circuit structure.

        Returns a function that takes (x, arch_samples) where:
          x: input data tensor
          arch_samples: list of one-hot tensors from sample_all()

        The function builds the circuit on the device and returns
        measurement results.
        """

        def circuit_fn(x: torch.Tensor, arch_samples: List[torch.Tensor]) -> List[torch.Tensor]:
            n_qubits = self.n_qubits
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="X")

            param_offset = 0
            for layer_idx in range(self.n_layers):
                sample = arch_samples[layer_idx]
                ops, param_offset = self.layers[layer_idx].build(
                    sample, self.circuit_params, param_offset
                )
                for op_name, wires, p, _ in ops:
                    self.pool.apply(op_name, p, wires)

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return qml.QNode(circuit_fn, device, interface="torch")

    def build_fixed_circuit_fn(
        self,
        device: qml.Device,
    ) -> Callable:
        """Create a QNode using the derived (discrete) architecture.

        After search is complete, use this to evaluate the final architecture.
        """

        arch_indices = [l.derived_architecture() for l in self.layers]

        def circuit_fn(x: torch.Tensor) -> List[torch.Tensor]:
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="X")
            param_offset = 0
            for layer_idx in range(self.n_layers):
                indices = arch_indices[layer_idx]
                n_single = self.n_qubits
                pos = 0
                for q in range(self.n_qubits):
                    op_name = self.pool.gate_names[indices[pos]]
                    n_p = self.pool.gates[op_name]["params"]
                    p = self.circuit_params[param_offset: param_offset + n_p]
                    self.pool.apply(op_name, p, [q])
                    param_offset += n_p
                    pos += 1
                for q in range(self.n_qubits - 1):
                    op_name = self.pool.gate_names[indices[pos]]
                    n_p = self.pool.gates[op_name]["params"]
                    p = self.circuit_params[param_offset: param_offset + n_p]
                    self.pool.apply(op_name, p, [q, q + 1])
                    param_offset += n_p
                    pos += 1
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return qml.QNode(circuit_fn, device, interface="torch")
