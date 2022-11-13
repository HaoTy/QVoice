from typing import Callable, Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms import QAOA, VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import OptimizerResult, SciPyOptimizer
from qiskit.opflow import AerPauliExpectation, ExpectationFactory, StateFn
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from scipy.optimize import NonlinearConstraint


class InConstraintSolver:
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

        if in_cons_prob_lbound and isinstance(self.optimizer, SciPyOptimizer):
            self.optimizer._kwargs.update(
                {
                    "constraints": [
                        {
                            "type": "ineq",
                            "fun": self.get_in_cons_prob,
                        },
                        # NonlinearConstraint(
                        #     self.get_in_cons_prob, in_cons_prob_lbound, 1.0
                        # ),
                    ]
                }
            )

        self.iter = 0
        self.n = len(problem.variables)
        self.log_level = log_level
        self.log = {}
        self.optimal_solution = None

        if log_level > 0:
            self.log = {"energy": []}

        if log_level > 1:
            self.log.update({"in_cons_prob": []})

        if log_level > 2:
            self.log.update({"opt_sol_prob": [], "approx_ratio": []})

            self.optimal_solution = "".join(
                [
                    str(int(bit))
                    for bit in MinimumEigenOptimizer(NumPyMinimumEigensolver())
                    .solve(problem)
                    .x
                ]
            )
            print("Optimal Solution: ", self.optimal_solution)
            self.max_fval, self.min_fval, cum_fval = -np.inf, np.inf, 0

        self.feasible_solns, self.feasible_indices, self.objective_values = (
            [],
            [],
            {},
        )
        if log_level > 2 or use_in_cons_energy or in_cons_prob_lbound:

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
                        self.objective_values[bitstr] = fval

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
        if not self.use_in_cons_energy and isinstance(
            self.expectation, AerPauliExpectation
        ):
            expect_op = self.algorithm._circuit_sampler.convert(expect_op)
        energy = np.real(expect_op.eval())
        if self.log_level > 0:
            self.log["energy"].append(energy)
        return energy

    def get_state(
        self, parameters: np.ndarray
    ) -> Union[List[float], Dict[str, int], QuantumCircuit]:
        if self.parameters is not None and np.allclose(
            self.parameters, parameters, rtol=0, atol=1e-13
        ):
            return self.state

        self.parameters = parameters
        self.iter += 1
        if isinstance(self.expectation, AerPauliExpectation):
            circuit = self.algorithm.ansatz.bind_parameters(parameters)
            if (
                not self.use_in_cons_energy
                and not self.in_cons_prob_lbound
                and self.log_level < 3
            ):
                self.state = circuit
                return circuit

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
            and self.log_level < 3
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
                    if bitstr not in self.objective_values.keys():
                        self.objective_values[bitstr] = self.problem.objective.evaluate(
                            [int(x) for x in bitstr]
                        )
                    fval += prob * self.objective_values[bitstr]
                    if bitstr == self.optimal_solution:
                        opt_sol_prob = prob
        else:
            in_cons_state = {}
            for bitstr, amplitude in state.items():
                prob = amplitude**2
                bitstr = bitstr[::-1]
                if bitstr in self.feasible_solns:
                    in_cons_prob += prob
                    in_cons_state[bitstr[::-1]] = amplitude
                    if bitstr not in self.objective_values.keys():
                        self.objective_values[bitstr] = self.problem.objective.evaluate(
                            [int(x) for x in bitstr]
                        )
                    fval += prob * self.objective_values[bitstr]
                    if bitstr == self.optimal_solution:
                        opt_sol_prob = prob
            in_cons_state = {
                bitstr: amplitude / (in_cons_prob**0.5)
                for bitstr, amplitude in in_cons_state.items()
            }

        self.in_cons_prob = in_cons_prob
        if self.log_level > 1:
            self.log["in_cons_prob"].append(in_cons_prob)

        if self.log_level > 2:
            sense = self.problem.objective.sense.value
            approx_ratio = (
                sense
                * (
                    fval / in_cons_prob
                    - (self.max_fval if sense == 1 else self.min_fval)
                )
                / (self.min_fval - self.max_fval)
            )
            self.log["approx_ratio"].append(approx_ratio)
            self.log["opt_sol_prob"].append(opt_sol_prob)

        self.state = (
            in_cons_state
            if self.use_in_cons_energy
            else (
                circuit if isinstance(self.expectation, AerPauliExpectation) else state
            )
        )
        return self.state

    def get_in_cons_prob(self, parameters: np.ndarray) -> float:
        self.get_state(parameters)
        return self.in_cons_prob - self.in_cons_prob_lbound

    def minimize(
        self,
        fun: Callable,
        x0: np.ndarray,
        jac: Optional[Callable] = None,
        bounds: Optional[list] = None,
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
        plt.axhline(
            self.in_cons_prob_lbound,
            label="lower bound on in-cons prob",
            linestyle="--",
        )
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
            raise NotImplementedError("Grid search is only supported for QAOA")
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
            if self.algorithm.quantum_instance.is_statevector:
                state = state_fn.primitive.data
                in_cons_state = np.zeros(len(state), dtype=complex)
                in_cons_state[self.feasible_indices] = state[self.feasible_indices]
                in_cons_prob = np.sum(np.abs(in_cons_state * in_cons_state.conj()))
                in_cons_state = in_cons_state / (in_cons_prob**0.5)
                state_squared = np.abs(state * state.conj())
                for idx, bitstr in zip(self.feasible_indices, self.feasible_solns):
                    prob = state_squared[idx]
                    if bitstr not in self.objective_values.keys():
                        self.objective_values[bitstr] = self.problem.objective.evaluate(
                            [int(x) for x in bitstr]
                        )
                    fval += prob * self.objective_values[bitstr]
            else:
                state = state_fn.to_dict_fn().primitive
                for bitstr, amplitude in state.items():
                    prob = amplitude**2
                    bitstr = bitstr[::-1]
                    if bitstr in self.feasible_solns:
                        in_cons_prob += prob
                        in_cons_state[bitstr[::-1]] = amplitude
                        if bitstr not in self.objective_values.keys():
                            self.objective_values[bitstr] = self.problem.objective.evaluate(
                                [int(x) for x in bitstr]
                            )
                        fval += prob * self.objective_values[bitstr]

            sense = self.problem.objective.sense.value
            approx_ratio = (
                sense
                * (
                    fval / in_cons_prob
                    - (self.max_fval if sense == 1 else self.min_fval)
                )
                / (self.min_fval - self.max_fval)
            )
            in_cons_probs.append(in_cons_prob)
            approx_ratios.append(approx_ratio)
        self.grid_search = {
            "approx_ratio": approx_ratios,
            "in_cons_prob": in_cons_probs,
        }
        plt.figure()
        plt.scatter(in_cons_probs, approx_ratios, s=1, label="grid search")
        plt.axhline(
            y=self.approx_ratio_baseline,
            color="g",
            linestyle="-",
            linewidth=1,
            label="random feasible state",
        )
        if self.in_cons_prob_lbound is not None:
            plt.axvline(
                x=self.in_cons_prob_lbound,
                color="c",
                linestyle="-",
                linewidth=1,
                label="in-constraint prob. lower bound",
            )
        if self.log_level > 2:
            plt.scatter(
                self.log["in_cons_prob"],
                self.log["approx_ratio"],
                marker="x",
                alpha=0.75,
                label="optimizer trace",
                c=range(len(self.log["in_cons_prob"])),
                cmap="autumn",
                s=200,
            )
        plt.xlabel("In-constraint probability", fontsize=16)
        plt.ylabel("Approximation ratio", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.show()
        if filename is not None:
            plt.savefig(filename)
