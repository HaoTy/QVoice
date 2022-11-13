from qiskit_optimization.applications import VertexCover
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model
import networkx as nx
import numpy as np


class WeightedVertexCover(VertexCover):
    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a vertex cover instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the vertex cover instance.
        """
        mdl = Model(name="Vertex cover")
        n = self._graph.number_of_nodes()
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(n)}
        objective = mdl.sum(self._graph.nodes[i]["weight"] * x[i] for i in x)
        for w, v in self._graph.edges:
            mdl.add_constraint(x[w] + x[v] >= 1)
        mdl.minimize(objective)
        op = from_docplex_mp(mdl)
        return op


# Linear objective with linear constraints
# Cost of evaluating constraints is O(m)
def get_random_instance(num_var: int, seed: int) -> QuadraticProgram:
    g = nx.dense_gnm_random_graph(num_var, num_var * (num_var - 1) // 4, seed)
    weights = np.random.default_rng(seed).normal(1, 1e-4, num_var)
    nx.set_node_attributes(g, values={i: weights[i] for i in range(num_var)}, name="weight")
    return WeightedVertexCover(g).to_quadratic_program()
