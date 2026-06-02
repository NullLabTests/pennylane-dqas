import pennylane as qml


def strong_entangling_layer(weights, wires, n_qubits):
    """Standard strongly entangling layer."""
    for i in range(n_qubits):
        qml.RX(weights[i, 0], wires=wires[i])
        qml.RY(weights[i, 1], wires=wires[i])
        qml.RZ(weights[i, 2], wires=wires[i])
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    if n_qubits > 2:
        qml.CNOT(wires=[wires[-1], wires[0]])


def basic_entangling_layer(weights, wires, n_qubits):
    """Basic entangling layer."""
    for i in range(n_qubits):
        qml.RX(weights[i], wires=wires[i])
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def hardware_efficient_layer(weights, wires, n_qubits):
    """Hardware-efficient ansatz layer."""
    for i in range(n_qubits):
        qml.RY(weights[i, 0], wires=wires[i])
        qml.RZ(weights[i, 1], wires=wires[i])
    for i in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def alternating_layer(weights, wires, n_qubits):
    """Alternating RX-RY layer with CZ entanglement."""
    for i in range(n_qubits):
        qml.RX(weights[i, 0], wires=wires[i])
        qml.RY(weights[i, 1], wires=wires[i])
    for i in range(n_qubits - 1):
        qml.CZ(wires=[wires[i], wires[i + 1]])


layer_presets = {
    "strong_entangling": strong_entangling_layer,
    "basic_entangling": basic_entangling_layer,
    "hardware_efficient": hardware_efficient_layer,
    "alternating": alternating_layer,
}
