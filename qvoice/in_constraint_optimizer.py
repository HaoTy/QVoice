from typing import Callable, Iterable, List, Optional, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from qiskit.algorithms import VQE, NumPyMinimumEigensolver, QAOA
from qiskit.algorithms.optimizers import OptimizerResult, COBYLA
from qiskit.circuit.library import QAOAAnsatz
from qiskit.opflow import OperatorBase, StateFn, ExpectationFactory
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer


class InConstraintOptimizer:
    def __init__(
        self,
        algorithm: VQE,
        problem: QuadraticProgram,
        penalty: Optional[float] = None,
        use_in_cons_energy: Optional[bool] = True,
        in_cons_prob_lbound: Optional[float] = None,
        log_level: Optional[int] = 0,
    ) -> None:
        self.problem = problem
        self.algorithm = algorithm
        self.penalty = penalty
        self.operator, self.offset = (
            QuadraticProgramToQubo(penalty).convert(problem).to_ising()
        )
        if algorithm.expectation is None:
            self.expectation = ExpectationFactory.build(
                operator=self.operator,
                backend=algorithm.quantum_instance,
                include_custom=algorithm._include_custom,
            )
        else:
            self.expectation = algorithm.expectation

        self.use_in_cons_energy = use_in_cons_energy
        self.in_cons_prob_lbound = in_cons_prob_lbound
        self.optimizer = algorithm.optimizer
        algorithm.optimizer = self
        self.state = None
        self.parameters = None
        self.in_cons_prob = None

        if in_cons_prob_lbound and isinstance(self.optimizer, COBYLA):
            self.optimizer._kwargs |= {
                "constraints": {
                    "type": "ineq",
                    "fun": lambda x: self.get_in_cons_prob(x) - in_cons_prob_lbound,
                }
            }

        self.iter = 0
        self.n = len(problem.variables)
        self.log_level = log_level

        self.log = {}
        if log_level > 0:
            self.log = {"energy": []}

        if log_level > 1:
            self.log |= {"in_cons_prob": []}

        if log_level > 2:
            self.log |= {"opt_sol_prob": [], "approx_ratio": []}

            self.optimal_solution = "".join(
                [
                    str(int(bit))
                    for bit in MinimumEigenOptimizer(NumPyMinimumEigensolver())
                    .solve(problem)
                    .x
                ]
            )
            print("Optimal Solution: ", self.optimal_solution)
            print("Optimal Solution: ", MinimumEigenOptimizer(NumPyMinimumEigensolver(), 0).solve(problem).x)
            self.max_fval, self.min_fval, cum_fval = -np.inf, np.inf, 0

        if log_level > 2 or use_in_cons_energy or in_cons_prob_lbound:
            self.feasible_solns, self.feasible_indices = [], []

            for i in range(2**self.n):
                bitstr = format(i, f"0{self.n}b")[::-1]
                assignment = [int(x) for x in bitstr]
                if problem.is_feasible(assignment):
                    self.feasible_solns.append(bitstr)
                    self.feasible_indices.append(i)
                    if log_level > 2:
                        fval = problem.objective.evaluate(assignment)
                        self.max_fval = max(fval, self.max_fval)
                        self.min_fval = min(fval, self.min_fval)
                        cum_fval += fval

            if log_level > 2:
                self.approx_ratio_baseline = (
                    cum_fval / len(self.feasible_solns) - self.min_fval
                ) / (self.max_fval - self.min_fval)
                print(
                    "Approximation ratio of random guess: ",
                    self.approx_ratio_baseline,
                )

    def get_energy(self, parameters: np.ndarray) -> float:
        expect_op = (
            self.expectation.convert(StateFn(self.operator, is_measurement=True))
            .compose(StateFn(self.get_state(parameters)))
            .reduce()
        )

        # parameter_sets = np.reshape(parameters, (-1, len(parameters)))
        # # Create dict associating each parameter with the lists of parameterization values for it
        # param_bindings = dict(
        #     zip(self.algorithm.ansatz.parameters, parameter_sets.transpose().tolist())
        # )
        # sampled_expect_op = self.algorithm._circuit_sampler.convert(
        #     expect_op, params=param_bindings
        # )
        energy = np.real(expect_op.eval())
        if self.log_level > 0:
            self.log["energy"].append(energy)
        return energy

    def get_state(self, parameters: np.ndarray) -> Union[List[float], Dict[str, int]]:
        if self.parameters is not None and np.allclose(self.parameters, parameters):
            return self.state

        self.parameters = parameters
        self.iter += 1
        # state = self.algorithm._get_eigenstate(parameters)

        state_fn = self.algorithm._circuit_sampler.convert(
            StateFn(self.algorithm.ansatz.bind_parameters(parameters))
        ).eval()
        if self.algorithm.quantum_instance.is_statevector:
            state = state_fn.primitive.data
        else:
            state = state_fn.to_dict_fn().primitive

        if (
            not self.use_in_cons_energy
            and not self.in_cons_prob_lbound
            and self.log_level < 2
        ):
            self.state = state
            return state

        in_cons_prob, fval, opt_sol_prob = 0, 0, 0

        if self.algorithm.quantum_instance.is_statevector:
            in_cons_state = np.zeros(len(state), dtype=complex)
            in_cons_state[self.feasible_indices] = state[self.feasible_indices]
            in_cons_prob = np.sum(np.abs(in_cons_state * in_cons_state.conj()))
            in_cons_state = in_cons_state / (in_cons_prob**0.5)
            if self.log_level > 2:
                state_squared = np.abs(state * state.conj())
                for idx, bitstr in zip(self.feasible_indices, self.feasible_solns):
                    prob = state_squared[idx]
                    fval += prob * self.problem.objective.evaluate(
                        [int(x) for x in bitstr]
                    )
                    if bitstr == self.optimal_solution:
                        opt_sol_prob = prob
        else:  # TODO
            raise NotImplementedError("qasm simulator not yet supported")
            # for idx, amplitude in state.items():
            #     prob = amplitude**2
            #     if idx in self.feasible_solns:
            #         in_cons_prob += prob
            #         if idx[::-1] == self.optimal_solution:
            #             opt_sol_prob = prob

        self.in_cons_prob = in_cons_prob
        if self.log_level > 1:
            self.log["in_cons_prob"].append(in_cons_prob)

        if self.log_level > 2:
            approx_ratio = (fval / in_cons_prob - self.max_fval) / (
                self.min_fval - self.max_fval
            )
            self.log["approx_ratio"].append(approx_ratio)
            self.log["opt_sol_prob"].append(opt_sol_prob)
            print(
                f"{in_cons_prob = }  {opt_sol_prob = }  {approx_ratio = }  {opt_sol_prob / in_cons_prob}"
            )

        self.state = in_cons_state
        return in_cons_state if self.use_in_cons_energy else state

    def get_in_cons_prob(self, parameters: np.ndarray) -> float:
        self.get_state(parameters)
        return self.in_cons_prob

    def minimize(
        self,
        fun: Callable,
        x0: np.ndarray,
        jac: Callable | None = None,
        bounds: list | None = None,
    ) -> OptimizerResult:

        if callable(self.optimizer):
            return self.optimizer(self.get_energy, x0, jac, bounds)
        else:
            return self.optimizer.minimize(self.get_energy, x0, jac, bounds)

    def solve(self):
        result = self.algorithm.compute_minimum_eigenvalue(self.operator)
        print("Iterations: ", self.iter)
        return result

    def plot_log(self, filename: Optional[str] = None) -> None:
        plt.figure()
        for k, v in self.log.items():
            if len(v) > 0:
                if k in ["penalty", "norm"]:
                    max_value = max(v)
                    v = np.array(v) / max_value
                    k += f" / {max_value}"
                elif k == "energy":
                    v = (np.array(v) - min(v)) / (max(v) - min(v))
                    k = "energy (normalized)"
                plt.plot(range(len(v)), v, label=f"{k}")
        plt.xlabel("Iteration", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.show()
        if filename is not None:
            plt.savefig(filename)

    def plot_pareto_frontier(
        self,
        beta_steps: Optional[int] = 20,
        gamma_steps: Optional[int] = 20,
        filename: Optional[str] = None,
    ) -> None:
        if not isinstance(self.algorithm, QAOA):
            raise NotImplementedError("Currently only supports QAOAAnsatz")
        reps = self.algorithm._reps
        # opt_beta, opt_gamma = 0, 0
        # opt_obj = np.inf
        in_cons_probs, approx_ratios = [], []

        for params in np.array(
            np.meshgrid(
                *np.linspace([0] * reps, [2 * np.pi] * reps, beta_steps, False, axis=1),
                *np.linspace(
                    [0] * reps, [2 * np.pi] * reps, gamma_steps, False, axis=1
                ),
            )
        ).T.reshape(-1, 2 * reps):
            state_fn = self.algorithm._circuit_sampler.convert(
                StateFn(self.algorithm.ansatz.bind_parameters(params))
            ).eval()
            in_cons_prob, fval = 0, 0
            state = state_fn.primitive.data
            state = np.abs(state * state.conj())
            for i in self.feasible_indices:
                prob = state[i]
                in_cons_prob += prob
                fval += prob * self.problem.objective.evaluate(
                    [int(x) for x in format(i, f"0{self.n}b")[::-1]]
                )
            approx_ratios.append(
                (fval / in_cons_prob - self.min_fval) / (self.max_fval - self.min_fval)
            )
            in_cons_probs.append(in_cons_prob)
            # if np.allclose(in_cons_prob, len(feasible_indices) / 2**n):
            #     print(f"beta/pi: {beta / np.pi}, gamma/2pi: {gamma / np.pi / 2}, approx_ratio: {approx_ratios[-1]}")
        self.grid_search = {
            "approx_ratio": approx_ratios,
            "in_cons_prob": in_cons_probs
        }
        plt.figure()
        plt.scatter(in_cons_probs, approx_ratios, s=1, label="grid search")
        # plt.scatter(in_cons_probs, 2 * approx_ratio_baseline - approx_ratios, s=1)
        plt.axhline(
            y=self.approx_ratio_baseline,
            color="g",
            linestyle="-",
            linewidth=1,
            label="uniformly random feasible states",
        )
        if self.in_cons_prob_lbound is not None:
            plt.axvline(
                x=self.in_cons_prob_lbound,
                color="c",
                linestyle="-",
                linewidth=1,
                label="in-constraint lower bound"
            )
        if self.log_level > 2:
            plt.scatter(
                self.log["in_cons_prob"],
                self.log["approx_ratio"],
                marker="x",
                alpha=0.75,
                label="optimizer",
                c=range(len(self.log["in_cons_prob"])),
                cmap='autumn'
            )
        plt.xlabel("In-constraint probability", fontsize=16)
        plt.ylabel("Approximation ratio", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
