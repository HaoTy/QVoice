from typing import Callable, Iterable, List, Optional, Union, Dict
from qiskit_optimization import QuadraticProgram

from ..objectives.vqa_objective import VQAObjective


class CustomVQASolver:
    def __init__(
        self,
        algorithm: VQE,
        problem: QuadraticProgram,
        penalty: Optional[float] = None,
        objective: Optional[VQAObjective] = None,
        constraints:  Optional[VQAObjective | List[Optional[VQAObjective]]] = None,
        log: Optional[VQAObjective | List[Optional[VQAObjective]]] = None,
    ) -> None:
        pass