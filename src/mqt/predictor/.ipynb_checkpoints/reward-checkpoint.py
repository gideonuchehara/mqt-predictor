"""This module contains the functions to calculate the reward of a quantum circuit on a given device."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from mqt.bench.utils import calc_supermarq_features

if TYPE_CHECKING:
    from qiskit import QuantumCircuit, QuantumRegister, Qubit

    from mqt.bench.devices import Device

logger = logging.getLogger("mqt-predictor")

figure_of_merit = Literal["expected_fidelity", "critical_depth", "adjacency_matrix_sum"]


def crit_depth(qc: QuantumCircuit, precision: int = 10) -> float:
    """Calculates the critical depth of a given quantum circuit."""
    supermarq_features = calc_supermarq_features(qc)
    return cast(float, np.round(1 - supermarq_features.critical_depth, precision))


def expected_fidelity(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
    """Calculates the expected fidelity of a given quantum circuit on a given device.

    Arguments:
        qc: The quantum circuit to be compiled.
        device: The device to be used for compilation.
        precision: The precision of the returned value. Defaults to 10.

    Returns:
        The expected fidelity of the given quantum circuit on the given device.
    """
    res = 1.0
    for instruction, qargs, _cargs in qc.data:
        gate_type = instruction.name

        if gate_type != "barrier":
            assert len(qargs) in [1, 2]
            first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)

            if len(qargs) == 1:
                if gate_type == "measure":
                    specific_fidelity = device.get_readout_fidelity(first_qubit_idx)
                else:
                    specific_fidelity = device.get_single_qubit_gate_fidelity(gate_type, first_qubit_idx)
            else:
                second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
                specific_fidelity = device.get_two_qubit_gate_fidelity(gate_type, first_qubit_idx, second_qubit_idx)

            res *= specific_fidelity

    return cast(float, np.round(res, precision))


def calc_qubit_index(qargs: list[Qubit], qregs: list[QuantumRegister], index: int) -> int:
    """Calculates the global qubit index for a given quantum circuit and qubit index."""
    offset = 0
    for reg in qregs:
        if qargs[index] not in reg:
            offset += reg.size
        else:
            qubit_index: int = offset + reg.index(qargs[index])
            return qubit_index
    error_msg = f"Global qubit index for local qubit {index} index not found."
    raise ValueError(error_msg)


import copy
import numpy as np
# import retworkx as rx
import rustworkx as rx
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

def to_networkx(circuit_dag):
    """Returns a copy of the DAGCircuit in networkx format."""
    try:
        import networkx as nx
    except ImportError as ex:
        raise ImportError("Networkx is needed to use to_networkx(). It "
                          "can be installed with 'pip install networkx'") from ex
    G = nx.MultiDiGraph()
    for node in circuit_dag._multi_graph.nodes():
        G.add_node(node)
    for node_id in rx.topological_sort(circuit_dag._multi_graph):
        for source_id, dest_id, edge in circuit_dag._multi_graph.in_edges(node_id):
            G.add_edge(circuit_dag._multi_graph[source_id],
                       circuit_dag._multi_graph[dest_id],
                       wire=edge)
    return G



def get_biadjacency_candidate_matrix(static_circuit):
    """
	Get the biadjacency and candidate matrices of the simplified bipartite graph 
	from the quantum circuit by searching for connections between the input qubits
	and the output qubits of the quantum circuit.

    Args:
        Qiskit QuantumCircuit object

    Returns:
        (numpy.ndarray, numpy.ndarray): the biadjacency and candidate matrices of 
		the simplified bipartite graph corresponding to a quantum circuit
    """
	
    circuit = copy.deepcopy(static_circuit)
    circuit.remove_final_measurements() # remove all circuit final measurements and barriers
    
    # convert circuit to qiskit DAG
    circ_dag = circuit_to_dag(circuit)
    
    graph = to_networkx(circ_dag) # convert from Qiskit DAG to Networkx DAG
    roots = list(circ_dag.input_map.values())  # the roots or input nodes of the circuit
    terminals = list(circ_dag.output_map.values()) # the output nodes of the circuit
	
    # Initialize the biadjacency matrix of an empty quantum circuit
    biadjacency_matrix = np.zeros((len(roots), len(terminals)), dtype=int)
    # For each root-terminal pair, if there is a path from root to terminal,
    # then the entry corresponding to this root-terminal pair will be one otherwise zero.
    for i, root in enumerate(roots):
        for j, terminal in enumerate(terminals):
            if nx.has_path(graph, root, terminal):
                biadjacency_matrix[i][j] += 1

    # Calculate the candidate matrix from the biadjacency matrix
    candidate_matrix = np.ones((len(terminals), len(roots)), dtype=int) - biadjacency_matrix.transpose()

    return biadjacency_matrix, candidate_matrix

def adjacency_matrix_sum(qc: QuantumCircuit) -> float:
    """Calculates the sum of the adjacency matrix of the simplified DAG of a given 
    quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to be compiled.

    Returns:
        int: The sum of the adjacency matrix of the simplified DAG of the given 
        quantum circuit on the given device.
    """

    biadjacency_matrix, candidate_matrix = get_biadjacency_candidate_matrix(qc)

    return -float(np.sum(biadjacency_matrix))
