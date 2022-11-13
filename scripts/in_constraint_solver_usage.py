#!/usr/bin/env python
# coding: utf-8

# This notebook demonstrates typical usages of the `InConstraintSolver` in `QVoice`.

# First, set the number of variables/qubits and the seed for generating the problem

# In[1]:


n = 10
seed = 42


# Define a problem with `qiskit_optimization`, `qiskit_finance`, or `docplex.mp`

# In[2]:


import networkx as nx
from qiskit_optimization.applications import GraphPartition

problem = GraphPartition(
        nx.random_partition_graph([n // 2] * 2, 1, 2 / n, seed)
    ).to_quadratic_program()

# from qiskit_finance.applications.optimization import PortfolioOptimization
# problem = PortfolioOptimization(...).to_quadratic_program()


# Or even easier, get a random instance of a supported problem from `qvoice.problems`

# In[3]:


from qvoice.problems import (
    portfolio_optimization,
    vertex_cover,
    clique,
    graph_partition,
    independent_set,
    max_bisection
)

problem = portfolio_optimization.get_random_instance(n, seed)


# Print out the problem formulation

# In[4]:


print(problem.prettyprint())


# Define the VQA algorithm with `qiskit` as usual

# In[5]:


import qvoice
from qiskit_aer import AerSimulator
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal


reps = 1
maxiter = 300
optimizer = COBYLA(maxiter=maxiter, disp=True)
backend = AerSimulator(method="statevector")

algorithm = VQE(
    TwoLocal(n, "ry", "cz", reps=reps, entanglement="linear"),
    optimizer=optimizer,
    quantum_instance=backend,
)
# or
algorithm = QAOA(
    reps=1,
    optimizer=optimizer,
    quantum_instance=backend,
)


# The `InConstraintSolver` can be constructed with one line of code using normal `qiskit` problem and VQA objects: 
# ```
# solver = qvoice.InConstraintSolver(algorithm, problem)
# ```

# It also can be fine-tuned with several options. 
# - `in_cons_prob_lbound` specifies the lower bound constraint for the in-constraint probability. Default is `None`.
# - `penalty` defines the penalty factor that controls the penalty term in the Hamiltonian. Default is `None`, which uses `qiskit_optimization`'s penalty heuristic: `qiskit_optimization.converters.LinearEqualityToPenalty()._auto_define_penalty(problem)`.
# - `use_in_cons_energy` can be set to `False` to use the normal penalized energy approach while still having metrics logged during the optimization process. Default is `True`.
# - `log_level` tells the solver what metrics to compute and log at each iteration. Higher `log_level` gives more information but also leads to greater overheads. Note that for some solver settings, certain metrics need to be computed regardless of how `log_level` specifies. Default is 0.
#     - `log_level=0` Nothing
#     - `log_level=1` energy
#     - `log_level=2` in-constraint probability
#     - `log_level=3` approximation ratio and optimal solution probability

# In[7]:


in_cons_prob_lbound = 0.05
penalty = None
solver = qvoice.InConstraintSolver(
    algorithm,
    problem,
    penalty=penalty,
    use_in_cons_energy=True,
    in_cons_prob_lbound=in_cons_prob_lbound,
    log_level=3,
)
print(solver.solve())


# Plot the logged metrics and the grid search overlaid with the optimizer trace.

# In[9]:


solver.plot_log()

beta_steps, gamma_steps = 30, 30
solver.plot_pareto_frontier(
    beta_steps=beta_steps,
    gamma_steps=gamma_steps,
)
