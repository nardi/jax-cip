"""Saving a compiled problem to a file and loading it in another process.

The file holds the compiled programs of every term and of the affine remainders, the
patterns of block terms, the bounds and the options. It does not hold the functions of the
problem or the trees of its variables and parameters. The caller gives the trees when
loading, which makes a loaded problem work exactly like a compiled one, and the file only
keeps the path and shape of each leaf to check them against.
"""

import importlib.metadata
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import cyxla
import jax
import numpy as np
from jaxtyping import PyTree

from jaxcip import _cache
from jaxcip._analysis import Analysis, InputRef, Linearity, Origin, RowSource, Term
from jaxcip._block import BlockHandle
from jaxcip._compile import (
    AffineEvaluator,
    Block,
    CompiledProblem,
    CompiledTerm,
    ParamsLayout,
    VariableLayout,
    compiled_terms,
    leaf_shape,
    remember_handle,
)
from jaxcip._handler import FunctionHandle
from jaxcip._options import CompileOptions
from jaxcip._problem_types import is_variable
from jaxcip._programs import TermInput, TermSignature
from jaxcip._types import Domain, Programs, Sense

KIND = "jaxcip.problem"
"""The kind of bundle that holds a compiled problem."""

VERSION = 1
"""The version of the payload, which changes when the data in the file changes."""

NO_JAXPR = jax.make_jaxpr(lambda: ())().jaxpr
"""The jaxpr of a term that was loaded, which only compilation needs."""


def dumps(compiled: CompiledProblem) -> bytes:
    """Returns a compiled problem as bytes that `loads` reads.

    The programs are stored together with their StableHLO, so a machine that cannot use the
    stored executables compiles the source instead.
    """
    writer, header = pack(compiled)
    return writer.to_bytes(header)


def save(compiled: CompiledProblem, path: str | os.PathLike[str]) -> None:
    """Saves a compiled problem to a file that `load` reads. `.jcp` is the usual extension."""
    writer, header = pack(compiled)
    writer.write(path, header)


def loads(
    data: bytes,
    variables: PyTree[Any],
    params: PyTree[Any] = None,
    *,
    on_mismatch: Literal["recompile", "raise"] = "recompile",
    warm_up: bool = True,
    threads: int | None = None,
) -> CompiledProblem:
    """Returns the compiled problem that `dumps` made from bytes.

    `variables` and `params` are trees with the structure that the problem was compiled for.
    The leaves of `variables` are `Variable`s or arrays, and those of `params` are arrays or
    `jax.ShapeDtypeStruct`s, as for `compile`. Only their paths and shapes are checked
    against the file, and the bounds and domains come from the file. The result solves like
    a compiled problem, with the structure of `variables` for `x`. Its `problem` is `None`,
    and its terms have no jaxprs.

    Executables that were stored for another machine are replaced by a compilation of the
    StableHLO, unless `on_mismatch` is `raise`. With `warm_up` each program runs once, so the
    first solve is as fast as the later ones. The executables load on `threads` threads, or
    one per CPU core.
    """
    return read(
        cyxla.Bundle(data, KIND, VERSION),
        variables,
        params,
        on_mismatch,
        warm_up,
        threads,
    )


def load(
    path: str | os.PathLike[str],
    variables: PyTree[Any],
    params: PyTree[Any] = None,
    *,
    on_mismatch: Literal["recompile", "raise"] = "recompile",
    warm_up: bool = True,
    threads: int | None = None,
) -> CompiledProblem:
    """Returns the compiled problem that `save` wrote to a file, as `loads` does."""
    return read(
        cyxla.Bundle.read(path, KIND, VERSION),
        variables,
        params,
        on_mismatch,
        warm_up,
        threads,
    )


class Packer:
    """Adds the pieces of a compiled problem to a bundle and describes them in a header."""

    def __init__(self, writer: cyxla.BundleWriter) -> None:
        self.writer = writer
        self.handles: list[dict[str, Any]] = []
        """The entries of the handles that were added, shared by the terms that use them."""
        self.indices: dict[str, int] = {}
        """The position of each handle among `handles`, by key."""

    def array(self, name: str, values: np.ndarray) -> dict[str, Any]:
        """Adds an array as a blob and returns what is needed to read it back."""
        values = np.ascontiguousarray(values)
        self.writer.add(name, values)
        return {"blob": name, "dtype": values.dtype.name, "shape": list(values.shape)}

    def programs(self, name: str, programs: Programs) -> list[dict[str, Any] | None]:
        """Adds the executables of a handle and returns their records."""
        return [
            None
            if executable is None
            else cyxla.pack_program(self.writer, f"{name}.{field}", executable)
            for field, executable in zip(Programs._fields, programs, strict=True)
        ]

    def handle(self, term: CompiledTerm) -> int:
        """Adds the handle of a term unless it was added, and returns its position."""
        known = self.indices.get(term.key)
        if known is not None:
            return known
        handle = term.handle
        index = len(self.handles)
        name = f"handle{index}"
        entry: dict[str, Any] = {
            "key": term.key,
            "num_inputs": handle.num_inputs,
            "num_params": handle.num_params,
            "programs": self.programs(name, handle.programs),
        }
        if isinstance(handle, BlockHandle):
            entry["pattern"] = {
                "pointers": self.array(f"{name}.pointers", handle.pointers),
                "columns": self.array(f"{name}.columns", handle.columns),
            }
        else:
            entry["state_size"] = handle.state_size
            entry["enclosure_size"] = handle.enclosure_size
        self.handles.append(entry)
        self.indices[term.key] = index
        return index

    def term(self, name: str, term: CompiledTerm) -> dict[str, Any]:
        """Returns the entry of a term, which refers to its handle."""
        spec = term.term
        signature = term.signature
        gather = None
        if spec.gather is not None:
            gather = {
                "leaves": self.array(f"{name}.leaves", spec.gather.leaves),
                "entries": self.array(f"{name}.entries", spec.gather.entries),
            }
        return {
            "handle": self.handle(term),
            "signature": {
                "inputs": [
                    [list(item.shape), np.dtype(item.dtype).name, item.is_variable]
                    for item in signature.inputs
                ],
                "shape": list(signature.shape),
                "num_inputs": signature.num_inputs,
                "num_params": signature.num_params,
                "state_size": signature.state_size,
                "enclosure_size": signature.enclosure_size,
            },
            "inputs": [[ref.origin.name, ref.index] for ref in spec.inputs],
            "selections": [
                None if selection is None else list(selection)
                for selection in spec.selections
            ],
            "row_axes": list(spec.row_axes),
            "gather": gather,
        }

    def affine(self, name: str, affine: AffineEvaluator) -> dict[str, Any]:
        """Returns the entry of an affine evaluator."""
        entry: dict[str, Any] = {
            "rows": self.array(f"{name}.rows", affine.rows),
            "columns": self.array(f"{name}.columns", affine.columns),
            "program": None
            if affine.program is None
            else cyxla.pack_program(self.writer, f"{name}.program", affine.program),
        }
        if affine.constants is not None and affine.values is not None:
            entry["constants"] = self.array(f"{name}.constants", affine.constants)
            entry["values"] = self.array(f"{name}.values", affine.values)
        return entry

    def block(self, name: str, block: Block) -> dict[str, Any]:
        """Returns the entry of a block with its terms, bounds and affine remainder."""
        analysis = block.analysis
        return {
            "shape": list(analysis.shape),
            "cut_shapes": [list(shape) for shape in analysis.cut_shapes],
            "constants": [
                self.array(f"{name}.constant{index}", np.asarray(constant))
                for index, constant in enumerate(analysis.constants)
            ],
            "lower": self.array(f"{name}.lower", block.lower),
            "upper": self.array(f"{name}.upper", block.upper),
            "terms": [
                self.term(f"{name}.term{index}", term)
                for index, term in enumerate(block.terms)
            ],
            "affine": None
            if block.affine is None
            else self.affine(f"{name}.affine", block.affine),
        }


def pack(compiled: CompiledProblem) -> tuple[cyxla.BundleWriter, dict[str, Any]]:
    """Returns the bundle writer and the header for a compiled problem.

    The programs and arrays are added to the writer, and their entries go into the header.
    """
    writer = cyxla.BundleWriter(KIND, VERSION)
    packer = Packer(writer)
    variables = compiled.variables
    writer.add("variable_lower", np.ascontiguousarray(variables.lower, np.float64))
    writer.add("variable_upper", np.ascontiguousarray(variables.upper, np.float64))
    objective = (
        None
        if compiled.objective is None
        else packer.block("objective", compiled.objective)
    )
    constraints = [
        packer.block(f"constraint{index}", block)
        for index, block in enumerate(compiled.constraints)
    ]
    header = {
        "jaxcip": importlib.metadata.version("jaxcip"),
        "cyxla": cyxla.__version__,
        "options": cyxla.encode_record(compiled.options),
        "sense": int(compiled.sense),
        "variables": cyxla.encode_fingerprint(variables.fingerprint),
        "domains": [int(domain) for domain in variables.domains],
        "params": cyxla.encode_fingerprint(compiled.params.fingerprint),
        "handles": packer.handles,
        "objective": objective,
        "constraints": constraints,
    }
    return writer, header


def variable_layout_for(
    template: PyTree[Any],
    bundle: cyxla.Bundle,
    meta: dict[str, Any],
) -> VariableLayout:
    """Returns the layout of the variables tree given to `load`, with the saved bounds."""
    leaves, structure = jax.tree.flatten(template, is_leaf=is_variable)
    fingerprint = cyxla.tree_fingerprint(template, leaf_shape, is_leaf=is_variable)
    cyxla.check_fingerprint(
        cyxla.decode_fingerprint(meta["variables"]), fingerprint, "variables"
    )
    shapes = tuple(leaf_shape(leaf) for leaf in leaves)
    sizes = [int(np.prod(shape, dtype=np.int64)) for shape in shapes]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    return VariableLayout(
        structure=structure,
        shapes=shapes,
        offsets=tuple(int(offset) for offset in offsets),
        lower=bundle.array("variable_lower", np.float64).copy(),
        upper=bundle.array("variable_upper", np.float64).copy(),
        domains=tuple(Domain(value) for value in meta["domains"]),
        fingerprint=fingerprint,
    )


def params_layout_for(template: PyTree[Any], meta: dict[str, Any]) -> ParamsLayout:
    """Returns the layout of the parameters tree given to `load`."""
    leaves, structure = jax.tree.flatten(template)
    avals = tuple(as_aval(leaf) for leaf in leaves)
    fingerprint = cyxla.tree_fingerprint(template, lambda leaf: as_aval(leaf).shape)
    cyxla.check_fingerprint(
        cyxla.decode_fingerprint(meta["params"]), fingerprint, "parameters"
    )
    return ParamsLayout(structure, avals, fingerprint)


def as_aval(leaf: Any) -> jax.ShapeDtypeStruct:
    """Returns the shape and type of a parameter leaf."""
    if isinstance(leaf, jax.ShapeDtypeStruct):
        return leaf
    array = np.asarray(leaf)
    return jax.ShapeDtypeStruct(array.shape, array.dtype)


class Reader:
    """Rebuilds the pieces of a compiled problem from a bundle."""

    def __init__(
        self,
        bundle: cyxla.Bundle,
        on_mismatch: Literal["recompile", "raise"],
        warm_up: bool,
        threads: int,
    ) -> None:
        self.bundle = bundle
        self.on_mismatch = on_mismatch
        self.warm_up = warm_up
        self.threads = threads
        self.statistics = _cache.BuildStatistics()
        self.handles: list[FunctionHandle | BlockHandle] = []

    def array(self, entry: dict[str, Any]) -> np.ndarray:
        """Returns an array that `Packer.array` described, as a copy."""
        values = self.bundle.array(entry["blob"], np.dtype(entry["dtype"]))
        return values.reshape(entry["shape"]).copy()

    def executables(
        self, records: Sequence[dict[str, Any] | None]
    ) -> list[cyxla.Executable | None]:
        """Loads the executables of records, on several threads, and warms them up."""
        present = [record for record in records if record is not None]
        if self.threads > 1 and len(present) > 1:
            with ThreadPoolExecutor(self.threads, "jaxcip-load") as pool:
                loaded = list(
                    pool.map(
                        lambda record: cyxla.load_program(
                            self.bundle, record, self.on_mismatch
                        ),
                        present,
                    )
                )
        else:
            loaded = [
                cyxla.load_program(self.bundle, record, self.on_mismatch)
                for record in present
            ]
        results = iter(loaded)
        executables: list[cyxla.Executable | None] = []
        for record in records:
            if record is None:
                executables.append(None)
                continue
            executable, built = next(results)
            self.statistics = self.statistics.add(
                _cache.BuildStatistics(
                    compile_seconds=built.compile_seconds,
                    load_seconds=built.load_seconds,
                    compiled=built.compiled,
                    loaded=built.loaded,
                )
            )
            if self.warm_up:
                executable.warm_up()
            executables.append(executable)
        return executables

    def load_handles(self, entries: list[dict[str, Any]]) -> None:
        """Rebuilds the handles, reusing those that this process already has by key."""
        missing = [entry for entry in entries if entry["key"] not in compiled_terms]
        records = [record for entry in missing for record in entry["programs"]]
        executables = iter(self.executables(records))
        created: dict[str, FunctionHandle | BlockHandle] = {}
        for entry in missing:
            programs = Programs._make([next(executables) for _ in entry["programs"]])
            handle: FunctionHandle | BlockHandle
            if "pattern" in entry:
                handle = BlockHandle(
                    entry["num_inputs"],
                    entry["num_params"],
                    self.array(entry["pattern"]["pointers"]),
                    self.array(entry["pattern"]["columns"]),
                    programs,
                )
            else:
                handle = FunctionHandle(
                    entry["num_inputs"],
                    entry["num_params"],
                    programs,
                    entry["state_size"],
                    entry["enclosure_size"],
                )
            created[entry["key"]] = handle
            remember_handle(entry["key"], handle)
        self.statistics = self.statistics._replace(
            reused=self.statistics.reused + len(entries) - len(missing)
        )
        self.handles = [
            created.get(entry["key"]) or compiled_terms[entry["key"]]
            for entry in entries
        ]

    def term(self, entry: dict[str, Any]) -> CompiledTerm:
        """Returns a term with its handle and without a jaxpr."""
        signature = entry["signature"]
        gather = None
        if entry["gather"] is not None:
            gather = RowSource(
                self.array(entry["gather"]["leaves"]),
                self.array(entry["gather"]["entries"]),
            )
        term = Term(
            jaxpr=NO_JAXPR,
            signs=(),
            inputs=tuple(
                InputRef(Origin[name], index) for name, index in entry["inputs"]
            ),
            selections=tuple(
                None if selection is None else tuple(selection)
                for selection in entry["selections"]
            ),
            linearity=Linearity.NONLINEAR,
            gather=gather,
            row_axes=tuple(entry["row_axes"]),
        )
        return CompiledTerm(
            term,
            TermSignature(
                inputs=tuple(
                    TermInput(tuple(shape), np.dtype(dtype), is_variable)
                    for shape, dtype, is_variable in signature["inputs"]
                ),
                shape=tuple(signature["shape"]),
                num_inputs=signature["num_inputs"],
                num_params=signature["num_params"],
                state_size=signature["state_size"],
                enclosure_size=signature["enclosure_size"],
            ),
            self.handles[entry["handle"]],
            self.keys[entry["handle"]],
        )

    keys: list[str]
    """The key of each handle, in the order of the entries."""

    def affine(self, entry: dict[str, Any]) -> AffineEvaluator:
        """Returns the evaluator of an affine remainder, with its program loaded."""
        (program,) = self.executables([entry["program"]])
        fixed = "constants" in entry
        return AffineEvaluator(
            self.array(entry["rows"]),
            self.array(entry["columns"]),
            program,
            self.array(entry["constants"]) if fixed else None,
            self.array(entry["values"]) if fixed else None,
        )

    def block(self, entry: dict[str, Any]) -> Block:
        """Returns a block, whose analysis only has what solving needs."""
        analysis = Analysis(
            shape=tuple(entry["shape"]),
            constants=tuple(self.array(item) for item in entry["constants"]),
            affine=None,
            nonlinear=(),
            cut_shapes=tuple(tuple(shape) for shape in entry["cut_shapes"]),
        )
        return Block(
            analysis=analysis,
            terms=tuple(self.term(item) for item in entry["terms"]),
            lower=self.array(entry["lower"]),
            upper=self.array(entry["upper"]),
            affine=None if entry["affine"] is None else self.affine(entry["affine"]),
        )


def read(
    bundle: cyxla.Bundle,
    variables: PyTree[Any],
    params: PyTree[Any],
    on_mismatch: Literal["recompile", "raise"],
    warm_up: bool,
    threads: int | None,
) -> CompiledProblem:
    """Builds a compiled problem from a bundle and the trees of its variables and parameters."""
    start = time.perf_counter()
    meta = bundle.meta
    blocks = [meta["objective"], *meta["constraints"]]
    records = [record for entry in meta["handles"] for record in entry["programs"]]
    records += [
        block["affine"]["program"]
        for block in blocks
        if block is not None and block["affine"] is not None
    ]
    needed = cyxla.record_custom_calls([record for record in records if record])
    cyxla.initialize(platform="cpu", jax_targets=bool(needed))

    variable_layout = variable_layout_for(variables, bundle, meta)
    params_layout = params_layout_for(params, meta)
    options = cyxla.decode_record(CompileOptions, meta["options"])
    reader = Reader(
        bundle, on_mismatch, warm_up, threads or options.threads or os.cpu_count() or 1
    )
    reader.keys = [entry["key"] for entry in meta["handles"]]
    reader.load_handles(meta["handles"])
    objective = None if meta["objective"] is None else reader.block(meta["objective"])
    constraints = tuple(reader.block(entry) for entry in meta["constraints"])
    return CompiledProblem(
        problem=None,
        options=options,
        sense=Sense(meta["sense"]),
        variables=variable_layout,
        params=params_layout,
        objective=objective,
        constraints=constraints,
        statistics=reader.statistics._replace(wall_seconds=time.perf_counter() - start),
    )
