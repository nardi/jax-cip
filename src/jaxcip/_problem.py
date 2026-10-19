"""A SCIP problem with variables, linear constraints and constraints on JAX functions.

This is the only module that creates SCIP instances. The Python layer describes a problem
once and builds a new `Scip` for every solve, so a `Scip` is filled, solved and read, but
never changed after solving.
"""

from pathlib import Path

import cython
import numpy as np
from cython.cimports.jaxcip._blas import (
    Dl_info,
    dladdr,
    lbt_forward,
    openblas_set_num_threads,
)
from cython.cimports.jaxcip._handler import (
    ExprData,
    FunctionHandle,
    create_expression,
    free_expr_data,
    include_handler,
    new_expr_data,
    switches_from,
)
from cython.cimports.jaxcip._scip import (
    SCIP,
    SCIP_CONS,
    SCIP_EXPR,
    SCIP_OKAY,
    SCIP_PARAM,
    SCIP_RETCODE,
    SCIP_SOL,
    SCIP_VAR,
    SCIP_Bool,
    SCIP_Longint,
    SCIP_Real,
    SCIPaddCons,
    SCIPaddOrigObjoffset,
    SCIPaddVar,
    SCIPcheckSolOrig,
    SCIPcreate,
    SCIPcreateConsBasicLinear,
    SCIPcreateConsBasicNonlinear,
    SCIPcreateExprSum,
    SCIPcreateExprVar,
    SCIPcreateOrigSol,
    SCIPcreateProbBasic,
    SCIPcreateVarBasic,
    SCIPfree,
    SCIPfreeSol,
    SCIPgetBestSol,
    SCIPgetBoolParam,
    SCIPgetCharParam,
    SCIPgetDualbound,
    SCIPgetGap,
    SCIPgetIntParam,
    SCIPgetLongintParam,
    SCIPgetNOrigVars,
    SCIPgetNParams,
    SCIPgetNSols,
    SCIPgetNTotalNodes,
    SCIPgetOrigVars,
    SCIPgetParam,
    SCIPgetParams,
    SCIPgetPrimalbound,
    SCIPgetRealParam,
    SCIPgetSolOrigObj,
    SCIPgetSolVal,
    SCIPgetSolvingTime,
    SCIPgetStatus,
    SCIPgetStringParam,
    SCIPincludeDefaultPlugins,
    SCIPinfinity,
    SCIPmajorVersion,
    SCIPminorVersion,
    SCIPparamGetDesc,
    SCIPparamGetName,
    SCIPparamGetType,
    SCIPreadProb,
    SCIPreleaseCons,
    SCIPreleaseExpr,
    SCIPreleaseVar,
    SCIPsetBoolParam,
    SCIPsetCharParam,
    SCIPsetEmphasis,
    SCIPsetIntParam,
    SCIPsetLongintParam,
    SCIPsetObjsense,
    SCIPsetRealParam,
    SCIPsetSolVals,
    SCIPsetStringParam,
    SCIPsolve,
    SCIPtechVersion,
    SCIPvarGetName,
    SCIPwriteOrigProblem,
)
from cython.cimports.libc.stdlib import free, malloc

from jaxcip._options import HandlerOptions
from jaxcip._types import (
    Domain,
    Emphasis,
    JaxTerm,
    ParameterType,
    ScipVersion,
    Sense,
    Status,
)

ParameterValue = bool | int | float | str
"""A value of a SCIP parameter of any type."""


class ScipError(RuntimeError):
    """A SCIP call returned an error code."""


@cython.cfunc
@cython.exceptval(-1)
def check(retcode: SCIP_RETCODE, action: str) -> cython.int:
    """Raises a `ScipError` naming the failed action unless SCIP returned `SCIP_OKAY`."""
    if retcode != SCIP_OKAY:
        raise ScipError(f"SCIP failed to {action} with return code {retcode}")
    return 0


def forward_blas() -> int:
    """Forwards the linked OpenBLAS to libblastrampoline and returns the number of forwarded symbols.

    Ipopt calls BLAS and LAPACK through libblastrampoline, which has no backend of its own.
    The library path comes from the loaded OpenBLAS, so it does not depend on environment
    variables or install locations.
    """
    info: Dl_info
    dladdr(cython.cast(cython.p_void, openblas_set_num_threads), cython.address(info))
    return lbt_forward(info.dli_fname, 1, 0, cython.NULL)


def scip_version() -> ScipVersion:
    """Returns the version of the linked SCIP library."""
    return ScipVersion(
        major=SCIPmajorVersion(), minor=SCIPminorVersion(), technical=SCIPtechVersion()
    )


def parameter_table() -> tuple[tuple[str, ParameterType, str], ...]:
    """Returns the name, type and description of every parameter of SCIP with its default plugins."""
    instance = Scip("parameters")
    params: cython.pointer[cython.pointer[SCIP_PARAM]] = SCIPgetParams(instance.scip)
    table = []
    index: cython.int
    for index in range(SCIPgetNParams(instance.scip)):
        table.append(
            (
                SCIPparamGetName(params[index]).decode(),
                ParameterType(SCIPparamGetType(params[index])),
                SCIPparamGetDesc(params[index]).decode(),
            )
        )
    return tuple(table)


@cython.cclass
class Scip:
    """One SCIP instance with the default plugins and the JAX expression handler."""

    scip: cython.pointer[SCIP]
    handles: list
    """The function handles of the problem's expressions, which must outlive the instance."""
    addresses: object
    """Addresses of the variables in the order they were added, as an array of `uintp`.

    SCIP keeps its own list of variables sorted by type, so indices refer to this one.
    """
    num_variables = cython.declare(cython.int, visibility="readonly")
    """The number of variables."""
    num_constraints: cython.int

    def __cinit__(self, name: str) -> None:
        self.scip = cython.NULL
        self.handles = []
        self.addresses = np.empty(16, dtype=np.uintp)
        self.num_variables = 0
        self.num_constraints = 0
        check(SCIPcreate(cython.address(self.scip)), "create an instance")
        check(SCIPincludeDefaultPlugins(self.scip), "include the default plugins")
        check(include_handler(self.scip), "include the JAX expression handler")
        check(SCIPcreateProbBasic(self.scip, name.encode()), "create a problem")

    def __dealloc__(self) -> None:
        if self.scip != cython.NULL:
            SCIPfree(cython.address(self.scip))

    @staticmethod
    def read(path: Path) -> Scip:
        """Reads a problem file in any format SCIP reads, such as `.osil`, `.nl` or `.cip`."""
        instance = Scip(path.stem)
        check(
            SCIPreadProb(instance.scip, str(path).encode(), cython.NULL), f"read {path}"
        )
        count: cython.int = SCIPgetNOrigVars(instance.scip)
        variables: cython.pointer[cython.pointer[SCIP_VAR]] = SCIPgetOrigVars(
            instance.scip
        )
        instance.addresses = np.empty(max(count, 1), dtype=np.uintp)
        index: cython.int
        for index in range(count):
            instance.addresses[index] = cython.cast(cython.size_t, variables[index])
        instance.num_variables = count
        return instance

    def write(self, path: Path) -> None:
        """Writes the problem in the format of the file extension, with JAX expressions opaque."""
        check(
            SCIPwriteOrigProblem(self.scip, str(path).encode(), cython.NULL, False),
            f"write {path}",
        )

    @property
    def infinity(self) -> float:
        """SCIP's value for infinity, the bound of unbounded variables and sides."""
        return SCIPinfinity(self.scip)

    @cython.cfunc
    def clamp(self, value: cython.double) -> cython.double:
        """Maps values beyond SCIP's infinity to its infinity."""
        infinity: cython.double = SCIPinfinity(self.scip)
        return min(max(value, -infinity), infinity)

    @cython.cfunc
    def variable(self, index: cython.int) -> cython.pointer[SCIP_VAR]:
        """Returns the variable at an index, raising an `IndexError` if there is none."""
        if index < 0 or index >= self.num_variables:
            raise IndexError(f"There is no variable {index}")
        return cython.cast(
            cython.pointer[SCIP_VAR], cython.cast(cython.size_t, self.addresses[index])
        )

    @cython.cfunc
    def variable_pointers(self) -> cython.pointer[cython.pointer[SCIP_VAR]]:
        """Returns the variables in the order they were added."""
        return cython.cast(
            cython.pointer[cython.pointer[SCIP_VAR]],
            cython.cast(cython.size_t, self.addresses.ctypes.data),
        )

    @property
    def variable_names(self) -> tuple[str, ...]:
        """The names of the variables in the order of their indices."""
        variables: cython.pointer[cython.pointer[SCIP_VAR]] = self.variable_pointers()
        names = []
        index: cython.int
        for index in range(self.num_variables):
            names.append(SCIPvarGetName(variables[index]).decode())
        return tuple(names)

    def add_variable(
        self, lower: float, upper: float, objective: float, domain: Domain
    ) -> int:
        """Adds a variable and returns its index."""
        index: cython.int = self.num_variables
        var: cython.pointer[SCIP_VAR] = cython.NULL
        check(
            SCIPcreateVarBasic(
                self.scip,
                cython.address(var),
                f"x{index}".encode(),
                self.clamp(lower),
                self.clamp(upper),
                objective,
                domain.value,
            ),
            "create a variable",
        )
        retcode: SCIP_RETCODE = SCIPaddVar(self.scip, var)
        if retcode == SCIP_OKAY:
            if index == self.addresses.shape[0]:
                self.addresses = np.concatenate(
                    [self.addresses, np.empty(index, dtype=np.uintp)]
                )
            # The problem holds its own reference, so the address stays valid after releasing.
            self.addresses[index] = cython.cast(cython.size_t, var)
            self.num_variables += 1
        SCIPreleaseVar(self.scip, cython.address(var))
        check(retcode, "add a variable")
        return index

    def add_linear_constraint(
        self,
        indices: cython.int[::1],
        coefficients: cython.double[::1],
        lower: float,
        upper: float,
    ) -> None:
        """Adds `lower ≤ coefficients · x[indices] ≤ upper`."""
        count: cython.int = indices.shape[0]
        if coefficients.shape[0] != count:
            raise ValueError("A linear constraint needs one coefficient per variable")
        variables = self.variable_array(indices)
        vars_pointer: cython.pointer[cython.pointer[SCIP_VAR]] = cython.cast(
            cython.pointer[cython.pointer[SCIP_VAR]],
            cython.cast(cython.size_t, variables.ctypes.data),
        )
        cons: cython.pointer[SCIP_CONS] = cython.NULL
        check(
            SCIPcreateConsBasicLinear(
                self.scip,
                cython.address(cons),
                f"linear{self.num_constraints}".encode(),
                count,
                vars_pointer,
                cython.address(coefficients[0]) if count > 0 else cython.NULL,
                self.clamp(lower),
                self.clamp(upper),
            ),
            "create a linear constraint",
        )
        self.add_constraint(cons)

    def variable_array(self, indices: cython.int[::1]) -> np.ndarray:
        """Returns the addresses of the variables at the indices, as an array of `uintp`."""
        addresses = np.empty(indices.shape[0], dtype=np.uintp)
        position: cython.int
        for position in range(indices.shape[0]):
            addresses[position] = cython.cast(
                cython.size_t, self.variable(indices[position])
            )
        return addresses

    @cython.cfunc
    def add_constraint(self, cons: cython.pointer[SCIP_CONS]) -> cython.void:
        """Adds a constraint to the problem and releases it."""
        retcode: SCIP_RETCODE = SCIPaddCons(self.scip, cons)
        self.num_constraints += 1
        SCIPreleaseCons(self.scip, cython.address(cons))
        check(retcode, "add a constraint")

    @cython.cfunc
    def jax_expression(
        self, term: JaxTerm, options: HandlerOptions
    ) -> cython.pointer[SCIP_EXPR]:
        """Returns a new JAX expression over variable expressions of the term's inputs."""
        handle: FunctionHandle = term.handle
        params: cython.double[::1] = term.params
        inputs: cython.int[::1] = term.inputs
        if inputs.shape[0] != handle.num_inputs:
            raise ValueError(
                f"The function takes {handle.num_inputs} inputs, got {inputs.shape[0]}"
            )
        if params.shape[0] != handle.num_params:
            raise ValueError(
                f"The function takes {handle.num_params} parameters, got {params.shape[0]}"
            )
        num_inputs: cython.int = inputs.shape[0]
        children: cython.pointer[cython.pointer[SCIP_EXPR]] = cython.cast(
            cython.pointer[cython.pointer[SCIP_EXPR]],
            malloc(num_inputs * cython.sizeof(cython.pointer[SCIP_EXPR])),
        )
        created: cython.int = 0
        expr: cython.pointer[SCIP_EXPR] = cython.NULL
        data: cython.pointer[ExprData] = cython.NULL
        index: cython.int
        try:
            for index in range(num_inputs):
                check(
                    SCIPcreateExprVar(
                        self.scip,
                        cython.address(children[index]),
                        self.variable(inputs[index]),
                        cython.NULL,
                        cython.NULL,
                    ),
                    "create a variable expression",
                )
                created += 1
            data = new_expr_data(
                handle.function, cython.address(params[0]), switches_from(options)
            )
            if data == cython.NULL:
                raise MemoryError()  # pragma: no cover
            retcode: SCIP_RETCODE = create_expression(
                self.scip, cython.address(expr), data, children
            )
            if retcode != SCIP_OKAY:
                free_expr_data(data)  # pragma: no cover
            check(retcode, "create a JAX expression")
        finally:
            # The JAX expression captures its children, so the variable expressions are released.
            for index in range(created):
                SCIPreleaseExpr(self.scip, cython.address(children[index]))
            free(children)
        self.handles.append(handle)
        return expr

    def add_jax_constraint(
        self,
        terms: list,
        linear_indices: cython.int[::1],
        linear_coefficients: cython.double[::1],
        constant: float,
        lower: float,
        upper: float,
        options: HandlerOptions,
    ) -> None:
        """Adds `lower ≤ Σ c f(x[inputs], params) + linear · x[linear_indices] + constant ≤ upper`.

        Each entry of `terms` is a `JaxTerm` with the function `f`, its parameters and inputs
        and its coefficient `c`.
        """
        num_terms: cython.int = len(terms)
        num_linear: cython.int = linear_indices.shape[0]
        if linear_coefficients.shape[0] != num_linear:
            raise ValueError("A linear part needs one coefficient per variable")
        size: cython.int = num_terms + num_linear + 1
        children: cython.pointer[cython.pointer[SCIP_EXPR]] = cython.cast(
            cython.pointer[cython.pointer[SCIP_EXPR]],
            malloc(size * cython.sizeof(cython.pointer[SCIP_EXPR])),
        )
        coefficients: cython.p_double = cython.cast(
            cython.p_double, malloc(size * cython.sizeof(cython.double))
        )
        created: cython.int = 0
        sum_expr: cython.pointer[SCIP_EXPR] = cython.NULL
        cons: cython.pointer[SCIP_CONS] = cython.NULL
        index: cython.int
        try:
            for index in range(num_terms):
                term: JaxTerm = terms[index]
                children[created] = self.jax_expression(term, options)
                coefficients[created] = term.coefficient
                created += 1
            for index in range(num_linear):
                check(
                    SCIPcreateExprVar(
                        self.scip,
                        cython.address(children[created]),
                        self.variable(linear_indices[index]),
                        cython.NULL,
                        cython.NULL,
                    ),
                    "create a variable expression",
                )
                coefficients[created] = linear_coefficients[index]
                created += 1
            check(
                SCIPcreateExprSum(
                    self.scip,
                    cython.address(sum_expr),
                    created,
                    children,
                    coefficients,
                    constant,
                    cython.NULL,
                    cython.NULL,
                ),
                "create a sum expression",
            )
            check(
                SCIPcreateConsBasicNonlinear(
                    self.scip,
                    cython.address(cons),
                    f"jax{self.num_constraints}".encode(),
                    sum_expr,
                    self.clamp(lower),
                    self.clamp(upper),
                ),
                "create a nonlinear constraint",
            )
            self.add_constraint(cons)
        finally:
            if sum_expr != cython.NULL:
                SCIPreleaseExpr(self.scip, cython.address(sum_expr))
            for index in range(created):
                SCIPreleaseExpr(self.scip, cython.address(children[index]))
            free(children)
            free(coefficients)

    def set_sense(self, sense: Sense) -> None:
        """Sets whether to minimize or maximize the objective."""
        check(SCIPsetObjsense(self.scip, sense.value), "set the objective sense")

    def add_objective_offset(self, offset: float) -> None:
        """Adds a constant to the objective."""
        check(SCIPaddOrigObjoffset(self.scip, offset), "add an objective offset")

    def set_emphasis(self, emphasis: Emphasis) -> None:
        """Applies one of SCIP's predefined parameter settings."""
        check(
            SCIPsetEmphasis(self.scip, emphasis.value, True),
            f"set the {emphasis.name} emphasis",
        )

    @cython.cfunc
    def parameter_type(self, name: bytes) -> ParameterType:
        """Returns the type of a parameter, raising a `KeyError` if SCIP has no such parameter."""
        param: cython.pointer[SCIP_PARAM] = SCIPgetParam(self.scip, name)
        if param == cython.NULL:
            raise KeyError(f"SCIP has no parameter {name.decode()}")
        return ParameterType(SCIPparamGetType(param))

    def set_parameter(self, name: str, value: ParameterValue) -> None:
        """Sets a parameter, converting the value to the parameter's type."""
        key: bytes = name.encode()
        action = f"set {name} to {value!r}"
        match self.parameter_type(key):
            case ParameterType.BOOL:
                check(SCIPsetBoolParam(self.scip, key, bool(value)), action)
            case ParameterType.INT:
                check(SCIPsetIntParam(self.scip, key, int(value)), action)
            case ParameterType.LONGINT:
                check(SCIPsetLongintParam(self.scip, key, int(value)), action)
            case ParameterType.REAL:
                check(SCIPsetRealParam(self.scip, key, float(value)), action)
            case ParameterType.CHAR:
                text = str(value).encode()
                if len(text) != 1:
                    raise ValueError(f"{name} takes one character, got {value!r}")
                check(SCIPsetCharParam(self.scip, key, text[0]), action)
            case ParameterType.STRING:
                check(SCIPsetStringParam(self.scip, key, str(value).encode()), action)

    def get_parameter(self, name: str) -> ParameterValue:
        """Returns the current value of a parameter."""
        key: bytes = name.encode()
        action = f"get {name}"
        flag: SCIP_Bool = 0
        integer: cython.int = 0
        long_integer: SCIP_Longint = 0
        real: SCIP_Real = 0.0
        character: cython.char = 0
        string: cython.p_char = cython.NULL
        match self.parameter_type(key):
            case ParameterType.BOOL:
                check(SCIPgetBoolParam(self.scip, key, cython.address(flag)), action)
                return bool(flag)
            case ParameterType.INT:
                check(SCIPgetIntParam(self.scip, key, cython.address(integer)), action)
                return integer
            case ParameterType.LONGINT:
                check(
                    SCIPgetLongintParam(self.scip, key, cython.address(long_integer)),
                    action,
                )
                return long_integer
            case ParameterType.REAL:
                check(SCIPgetRealParam(self.scip, key, cython.address(real)), action)
                return real
            case ParameterType.CHAR:
                check(
                    SCIPgetCharParam(self.scip, key, cython.address(character)), action
                )
                return chr(character)
            case ParameterType.STRING:
                check(
                    SCIPgetStringParam(self.scip, key, cython.address(string)), action
                )
                return string.decode()

    def solve(self) -> None:
        """Solves the problem without holding the GIL."""
        retcode: SCIP_RETCODE
        with cython.nogil:
            retcode = SCIPsolve(self.scip)
        check(retcode, "solve")

    @property
    def status(self) -> Status:
        """Why SCIP stopped solving."""
        return Status(SCIPgetStatus(self.scip))

    @property
    def primal_bound(self) -> float:
        """The objective value of the best solution, or infinity in the direction of optimization."""
        return SCIPgetPrimalbound(self.scip)

    @property
    def dual_bound(self) -> float:
        """The proven bound on the optimal objective value."""
        return SCIPgetDualbound(self.scip)

    @property
    def gap(self) -> float:
        """The relative gap between the primal and the dual bound."""
        return SCIPgetGap(self.scip)

    @property
    def nodes(self) -> int:
        """The number of branch-and-bound nodes, including those of restarts."""
        return SCIPgetNTotalNodes(self.scip)

    @property
    def solving_time(self) -> float:
        """Seconds spent solving."""
        return SCIPgetSolvingTime(self.scip)

    def best_solution(self) -> tuple[float, np.ndarray] | None:
        """Returns the objective value and the variable values of the best solution, if any."""
        if SCIPgetNSols(self.scip) == 0:
            return None
        sol: cython.pointer[SCIP_SOL] = SCIPgetBestSol(self.scip)
        count: cython.int = self.num_variables
        variables: cython.pointer[cython.pointer[SCIP_VAR]] = self.variable_pointers()
        values = np.empty(count)
        view: cython.double[::1] = values
        index: cython.int
        for index in range(count):
            view[index] = SCIPgetSolVal(self.scip, sol, variables[index])
        return SCIPgetSolOrigObj(self.scip, sol), values

    def check_solution(self, values: cython.double[::1]) -> bool:
        """Returns whether an assignment of all variables satisfies every constraint and bound."""
        count: cython.int = self.num_variables
        if values.shape[0] != count:
            raise ValueError(
                f"The problem has {count} variables, got {values.shape[0]} values"
            )
        sol: cython.pointer[SCIP_SOL] = cython.NULL
        feasible: SCIP_Bool = 0
        check(
            SCIPcreateOrigSol(self.scip, cython.address(sol), cython.NULL),
            "create a solution",
        )
        try:
            if count > 0:
                check(
                    SCIPsetSolVals(
                        self.scip,
                        sol,
                        count,
                        self.variable_pointers(),
                        cython.address(values[0]),
                    ),
                    "set solution values",
                )
            check(
                SCIPcheckSolOrig(self.scip, sol, cython.address(feasible), False, True),
                "check a solution",
            )
        finally:
            SCIPfreeSol(self.scip, cython.address(sol))
        return bool(feasible)
