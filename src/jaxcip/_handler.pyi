import numpy as np
from jaxtyping import Float

from jaxcip._types import Estimator, HandlerStatistics, Interval, Program, Programs

HANDLER_NAME: bytes
PROGRAM_SLOTS: tuple[tuple[int, ...], ...]

def output_sizes(program: Program, num_inputs: int) -> tuple[int, ...]: ...
def check_length(name: str, values: object, expected: int) -> None: ...

class FunctionHandle:
    @property
    def num_inputs(self) -> int: ...
    @property
    def num_params(self) -> int: ...
    @property
    def programs(self) -> Programs: ...
    def __init__(
        self, num_inputs: int, num_params: int, programs: Programs
    ) -> None: ...
    def statistics(self) -> HandlerStatistics: ...
    def reset_statistics(self) -> None: ...
    def set_input(self, slot: int, values: Float[np.ndarray, " n"]) -> None: ...
    def evaluate(
        self, x: Float[np.ndarray, " n"], params: Float[np.ndarray, " m"]
    ) -> float: ...
    def value_and_gradient(
        self, x: Float[np.ndarray, " n"], params: Float[np.ndarray, " m"]
    ) -> tuple[float, Float[np.ndarray, " n"]]: ...
    def hessian_vector(
        self,
        x: Float[np.ndarray, " n"],
        direction: Float[np.ndarray, " n"],
        params: Float[np.ndarray, " m"],
    ) -> Float[np.ndarray, " n"]: ...
    def estimator(
        self,
        reference: Float[np.ndarray, " n"],
        lower: Float[np.ndarray, " n"],
        upper: Float[np.ndarray, " n"],
        params: Float[np.ndarray, " m"],
        overestimate: bool,
    ) -> Estimator: ...
    def interval(
        self,
        lower: Float[np.ndarray, " n"],
        upper: Float[np.ndarray, " n"],
        params: Float[np.ndarray, " m"],
    ) -> Interval: ...
    def gradient_interval(
        self,
        lower: Float[np.ndarray, " n"],
        upper: Float[np.ndarray, " n"],
        params: Float[np.ndarray, " m"],
    ) -> Interval: ...
    def time_program(
        self,
        program: Program,
        points: Float[np.ndarray, "count n"],
        lower: Float[np.ndarray, " n"],
        upper: Float[np.ndarray, " n"],
        params: Float[np.ndarray, " m"],
        repeat: int = 1,
    ) -> float: ...
