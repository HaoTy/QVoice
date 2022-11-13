from qiskit_optimization.applications import Clique
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model
import networkx as nx
import numpy as np


class WeightedClique(Clique):
    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a clique problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`.
        When "size" is None, this makes an optimization model for a maximal clique
        instead of the specified size of a clique.

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the clique problem instance.
        """
        complement_g = nx.complement(self._graph)

        mdl = Model(name="Clique")
        n = self._graph.number_of_nodes()
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(n)}
        for w, v in complement_g.edges:
            mdl.add_constraint(x[w] + x[v] <= 1)
        if self.size is None:
            mdl.maximize(mdl.sum(self._graph.nodes[i]["weight"] * x[i] for i in x))
        else:
            mdl.add_constraint(mdl.sum(x[i] for i in x) == self.size)
        op = from_docplex_mp(mdl)
        return op


# Linear objective with linear constraints
# Cost of evaluating constraints is O(m)
def get_random_instance(num_var: int, seed: int) -> QuadraticProgram:
    g = nx.dense_gnm_random_graph(num_var, num_var * (num_var - 1) // 4, seed)
    weights = np.random.default_rng(seed).normal(1, 1e-4, num_var)
    nx.set_node_attributes(g, values={i: weights[i] for i in range(num_var)}, name="weight")
    return WeightedClique(g).to_quadratic_program()
