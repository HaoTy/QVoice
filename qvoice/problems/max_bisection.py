from qiskit_optimization.applications import Maxcut
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model
import networkx as nx
import numpy as np


class WeightedMaxBisection(Maxcut):
    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a Max-cut problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the Max-cut problem instance.
        """
        mdl = Model(name="Max-bisection")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(self._graph.number_of_nodes())}
        objective = mdl.sum(
            self._graph.edges[i, j]["weight"] * x[i] * (1 - x[j])
            + self._graph.edges[i, j]["weight"] * x[j] * (1 - x[i])
            for i, j in self._graph.edges
        )
        mdl.add_constraint(mdl.sum([x[i] for i in x]) == self._graph.number_of_nodes() // 2)
        mdl.maximize(objective)
        op = from_docplex_mp(mdl)
        return op


# Quatratic objective with linear constraints
# Cost of evaluating constraints is O(n)
def get_random_instance(num_var: int, seed: int) -> QuadraticProgram:
    g = nx.random_regular_graph(3, num_var, seed)
    weights = np.random.default_rng(seed).normal(1, 1e-4, 3 * num_var // 2)
    for i, (w, v) in enumerate(g.edges):
        g.edges[w, v]["weight"] = weights[i]
    return WeightedMaxBisection(g).to_quadratic_program()
