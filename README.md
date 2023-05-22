# QVoice: Quantum Variational Optimization with In-Constraint Probability
This is a package accompanying the paper [Exploiting In-Constraint Energy in Constrained Variational Quantum Optimization](https://arxiv.org/abs/2211.07016). QVoice runs variational quantum algorithms with in-constraint energy, an alternative objective for constrained problems, to achieve higher accuracies and lower numbers of iterations in comparison to using energy (exepectation value of the Hamiltonian) as the objective.

## Installation
```bash
git clone https://github.com/haoty/QVoice.git
pip install -e ./QVoice
```

## Get Started
### Minimal usage example
```python
import qvoice
from qiskit_aer import AerSimulator
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA

problem = qvoice.problems.clique.get_random_instance(num_var=8, seed=42)
algorithm = QAOA(optimizer=COBYLA(), quantum_instance=AerSimulator(method="statevector"))
result = qvoice.InConstraintSolver(algorithm, problem).solve()
```

### Walkthrough of functionalities provided by `InConstraintSolver`
__The following walkthrough is also available in the format of [Jupyter notebook](https://github.com/HaoTy/QVoice/blob/main/notebooks/in_constraint_solver_usage.ipynb) and [Python script](https://github.com/HaoTy/QVoice/blob/main/scripts/in_constraint_solver_usage.py).__

First, set the number of variables/qubits and the seed for generating the problem

```python
n = 10
seed = 42
```

Define a problem with `qiskit_optimization`, `qiskit_finance`, or `docplex.mp`


```python
import networkx as nx
from qiskit_optimization.applications import GraphPartition

problem = GraphPartition(
        nx.random_partition_graph([n // 2] * 2, 1, 2 / n, seed)
    ).to_quadratic_program()

# from qiskit_finance.applications.optimization import PortfolioOptimization
# problem = PortfolioOptimization(...).to_quadratic_program()
```

Or even easier, get a random instance of a supported problem from `qvoice.problems`


```python
from qvoice.problems import (
    portfolio_optimization,
    vertex_cover,
    clique,
    graph_partition,
    independent_set,
    max_bisection
)

problem = portfolio_optimization.get_random_instance(n, seed)
```

Print out the problem formulation


```python
print(problem.prettyprint())
```

Define the VQA algorithm with `qiskit` as usual


```python
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
```

The `InConstraintSolver` can be constructed with one line of code using normal `qiskit` problem and VQA objects

```python
solver = qvoice.InConstraintSolver(algorithm, problem)
```

It also can be fine-tuned with several options. 
- `in_cons_prob_lbound` specifies the lower bound constraint for the in-constraint probability. Default is `None`.
- `penalty` defines the penalty factor that controls the penalty term in the Hamiltonian. Default is `None`, which uses `qiskit_optimization`'s penalty heuristic: `qiskit_optimization.converters.LinearEqualityToPenalty()._auto_define_penalty(problem)`.
- `use_in_cons_energy` can be set to `False` to use the normal penalized energy approach while still having metrics logged during the optimization process. Default is `True`.
- `log_level` tells the solver what metrics to compute and log at each iteration. Higher `log_level` gives more information but also leads to greater overheads. Note that for some solver settings, certain metrics need to be computed regardless of how `log_level` specifies. Default is 0.
    - `log_level=0` Nothing
    - `log_level=1` energy
    - `log_level=2` in-constraint probability
    - `log_level=3` approximation ratio and optimal solution probability


```python
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
```

Plot the logged metrics and the grid search overlaid with the optimizer trace.


```python
solver.plot_log()

beta_steps, gamma_steps = 30, 30
solver.plot_pareto_frontier(
    beta_steps=beta_steps,
    gamma_steps=gamma_steps,
)

```


## Roadmap
### Doing
- Refactor to support custom objectives and constraints

### TO-DO
- Support lazy evaluation of function values.

### Longterm
- Integrate into Qiskit
