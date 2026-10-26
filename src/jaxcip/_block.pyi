import numpy as np
from jaxtyping import Float, Int

from jaxcip._options import HandlerOptions
from jaxcip._types import Estimator, HandlerStatistics, Interval, Program, Programs

BLOCK_HANDLER_NAME: bytes
BLOCK_SLOTS: tuple[tuple[int, ...], ...]

def block_output_sizes(program: Program, rows: int, nnz: int) -> tuple[int, ...]: ...
def check_length(name: str, values: object, expected: int) -> None: ...

class BlockHandle:
    @property
    def num_inputs(self) -> int: ...
    @property
    def num_params(self) -> int: ...
    @property
    def num_entries(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    @property
    def programs(self) -> Programs: ...
    @property
    def pointers(self) -> Int[np.ndarray, " rows_plus_one"]: ...
    @property
    def columns(self) -> Int[np.ndarray, " nnz"]: ...
    def __init__(
        self,
        num_inputs: int,
        num_params: int,
        pointers: Int[np.ndarray, " rows_plus_one"],
        columns: Int[np.ndarray, " nnz"],
        programs: Programs,
    ) -> None: ...
    def statistics(self) -> HandlerStatistics: ...
    def reset_statistics(self) -> None: ...
    def run(
        self,
        program: Program,
        x: Float[np.ndarray, " n"],
        params: Float[np.ndarray, " m"],
        direction: Float[np.ndarray, " n"] | None = None,
        lower: Float[np.ndarray, " n"] | None = None,
        upper: Float[np.ndarray, " n"] | None = None,
    ) -> tuple[Float[np.ndarray, " k"], ...]: ...
    def value_and_jacobian(
        self, x: Float[np.ndarray, " n"], params: Float[np.ndarray, " m"]
    ) -> tuple[Float[np.ndarray, " rows"], Float[np.ndarray, " nnz"]]: ...
    def hessian_vector(
        self,
        x: Float[np.ndarray, " n"],
        direction: Float[np.ndarray, " n"],
        params: Float[np.ndarray, " m"],
    ) -> Float[np.ndarray, " nnz"]: ...
    def relaxation(
        self,
        x: Float[np.ndarray, " n"],
        lower: Float[np.ndarray, " n"],
        upper: Float[np.ndarray, " n"],
        params: Float[np.ndarray, " m"],
        overestimate: bool = False,
    ) -> tuple[Float[np.ndarray, " rows"], Float[np.ndarray, " nnz"]]: ...
    def estimator(
        self,
        entry: int,
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

class BlockStateRef:
    @property
    def handle(self) -> BlockHandle: ...
    @property
    def inputs(self) -> Int[np.ndarray, " n"]: ...
    def __init__(
        self,
        handle: BlockHandle,
        params: Float[np.ndarray, " m"],
        inputs: Int[np.ndarray, " n"],
        options: HandlerOptions,
    ) -> None: ...
