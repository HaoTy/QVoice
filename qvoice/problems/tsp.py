import numpy as np
from qiskit_optimization.applications import Tsp
from qiskit_optimization import QuadraticProgram


# Note: the number of qubits is the number of edges, i.e. the number of vertices squared
# Quadratic objective with linear constraints
# Cost of evaluating constraints is O(n)
def get_random_instance(num_var: int, seed: int) -> QuadraticProgram:
    return Tsp.create_random_instance(
        int(np.sqrt(num_var)), seed=seed
    ).to_quadratic_program()
