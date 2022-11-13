from qiskit_optimization.applications import GraphPartition
from qiskit_optimization import QuadraticProgram
import networkx as nx


# A variant of the graph partition problem, where the constraint is to have two partitions with half the vertices each
# Quadratic objective with linear constraints
# Cost of evaluating constraints is O(n)
def get_random_instance(num_var: int, seed: int) -> QuadraticProgram:
    return GraphPartition(
        nx.random_partition_graph([num_var // 2] * 2, 1, 2 / num_var, seed)
    ).to_quadratic_program()
