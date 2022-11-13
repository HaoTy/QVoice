from qiskit_optimization.applications import StableSet
from qiskit_optimization import QuadraticProgram
import networkx as nx


# Linear objective with linear constraints
# Cost of evaluating constraints is O(m)
def get_random_instance(num_var: int, seed: int) -> QuadraticProgram:
    return StableSet(nx.dense_gnm_random_graph(num_var, num_var * (num_var - 1) // 4, seed)).to_quadratic_program()
