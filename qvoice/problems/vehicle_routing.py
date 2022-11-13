import numpy as np
from qiskit_optimization.applications import VehicleRouting
from qiskit_optimization import QuadraticProgram


# Note: the number of qubits is n(n-1), where n is the number of vertices
# Linear objective with linear constraints
# Cost of evaluating constraints is beyond O(n)?
def get_random_instance(num_var: int, seed: int) -> QuadraticProgram:
    return VehicleRouting.create_random_instance(
        int(np.sqrt(num_var)), seed=seed
    ).to_quadratic_program()
