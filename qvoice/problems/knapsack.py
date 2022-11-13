from qiskit_optimization.applications import Knapsack
from qiskit_optimization import QuadraticProgram
import numpy as np


# Needs auxiliary variables/qubits 
# Linear objective with linear constraint
# Cost of evaluating constraints is O(n)
def get_random_instance(
    num_var: int, seed: int
) -> QuadraticProgram:
    rng = np.random.default_rng(seed)
    weights = rng.integers(0, num_var, size=num_var) + num_var
    return Knapsack(
        values=rng.integers(0, num_var, size=num_var) + num_var,
        weights=weights,
        max_weight=np.sum(weights) // 2
    ).to_quadratic_program()
