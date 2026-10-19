"""Declarations of the part of SCIP's C API that the extension modules use.

Callback parameters are declared with the exact signatures of SCIP's `SCIP_DECL_EXPR*` macros,
so the C functions of `jaxcip._handler` can be passed to SCIP directly. Enumerations that
Python code chooses from are declared as plain integers, and `jaxcip._types` holds their
values.
"""


cdef extern from "scip/scip.h" nogil:
    ctypedef double SCIP_Real
    ctypedef unsigned int SCIP_Bool
    ctypedef long long SCIP_Longint

    ctypedef enum SCIP_RETCODE:
        SCIP_OKAY
        SCIP_ERROR
        SCIP_NOMEMORY

    ctypedef int SCIP_VARTYPE
    ctypedef int SCIP_OBJSENSE
    ctypedef int SCIP_STATUS
    ctypedef int SCIP_PARAMTYPE
    ctypedef int SCIP_PARAMEMPHASIS

    ctypedef struct SCIP
    ctypedef struct SCIP_SOL
    ctypedef struct SCIP_VAR
    ctypedef struct SCIP_CONS
    ctypedef struct SCIP_HEUR
    ctypedef struct SCIP_PARAM
    ctypedef struct SCIP_EXPR
    ctypedef struct SCIP_EXPRHDLR
    ctypedef struct SCIP_EXPRDATA
    ctypedef struct SCIP_EXPRHDLRDATA

    ctypedef struct SCIP_INTERVAL:
        SCIP_Real inf
        SCIP_Real sup

    ctypedef enum SCIP_MONOTONE:
        SCIP_MONOTONE_UNKNOWN
        SCIP_MONOTONE_INC
        SCIP_MONOTONE_DEC
        SCIP_MONOTONE_CONST

    ctypedef struct SYM_EXPRDATA "struct SYM_ExprData":
        SCIP_Real* constants
        int nconstants
        SCIP_Real* coefficients
        int ncoefficients
        SCIP_EXPR** children

    const SCIP_Real SCIP_INVALID
    const SCIP_Real SCIP_INTERVAL_INFINITY
    enum:
        SCIP_EXPR_MAXINITESTIMATES


# SCIP defines the callback types through macros, so Cython declares them itself with the same
# signatures.
ctypedef SCIP_RETCODE (*ExprCopyHdlr)(SCIP* scip, SCIP_EXPRHDLR* sourceexprhdlr) noexcept nogil
ctypedef SCIP_RETCODE (*ExprCopyData)(SCIP* targetscip, SCIP_EXPRHDLR* targetexprhdlr, SCIP_EXPRDATA** targetexprdata, SCIP* sourcescip, SCIP_EXPR* sourceexpr) noexcept nogil
ctypedef SCIP_RETCODE (*ExprFreeData)(SCIP* scip, SCIP_EXPR* expr) noexcept nogil
ctypedef int (*ExprCompare)(SCIP* scip, SCIP_EXPR* expr1, SCIP_EXPR* expr2) noexcept nogil
ctypedef SCIP_RETCODE (*ExprHash)(SCIP* scip, SCIP_EXPR* expr, unsigned int* hashkey, unsigned int* childrenhashes) noexcept nogil
ctypedef SCIP_RETCODE (*ExprEval)(SCIP* scip, SCIP_EXPR* expr, SCIP_Real* val, SCIP_SOL* sol) noexcept nogil
ctypedef SCIP_RETCODE (*ExprBwdiff)(SCIP* scip, SCIP_EXPR* expr, int childidx, SCIP_Real* val) noexcept nogil
ctypedef SCIP_RETCODE (*ExprFwdiff)(SCIP* scip, SCIP_EXPR* expr, SCIP_Real* dot, SCIP_SOL* direction) noexcept nogil
ctypedef SCIP_RETCODE (*ExprBwfwdiff)(SCIP* scip, SCIP_EXPR* expr, int childidx, SCIP_Real* bardot, SCIP_SOL* direction) noexcept nogil
ctypedef SCIP_INTERVAL (*ExprIntevalVar)(SCIP* scip, SCIP_VAR* var, void* intevalvardata) noexcept nogil
ctypedef SCIP_RETCODE (*ExprInteval)(SCIP* scip, SCIP_EXPR* expr, SCIP_INTERVAL* interval, ExprIntevalVar intevalvar, void* intevalvardata) noexcept nogil
ctypedef SCIP_RETCODE (*ExprEstimate)(SCIP* scip, SCIP_EXPR* expr, SCIP_INTERVAL* localbounds, SCIP_INTERVAL* globalbounds, SCIP_Real* refpoint, SCIP_Bool overestimate, SCIP_Real targetvalue, SCIP_Real* coefs, SCIP_Real* constant, SCIP_Bool* islocal, SCIP_Bool* success, SCIP_Bool* branchcand) noexcept nogil
ctypedef SCIP_RETCODE (*ExprInitEstimates)(SCIP* scip, SCIP_EXPR* expr, SCIP_INTERVAL* bounds, SCIP_Bool overestimate, SCIP_Real** coefs, SCIP_Real* constant, int* nreturned) noexcept nogil
ctypedef SCIP_RETCODE (*ExprMonotonicity)(SCIP* scip, SCIP_EXPR* expr, int childidx, SCIP_MONOTONE* result) noexcept nogil
ctypedef SCIP_RETCODE (*ExprReverseProp)(SCIP* scip, SCIP_EXPR* expr, SCIP_INTERVAL bounds, SCIP_INTERVAL* childrenbounds, SCIP_Bool* infeasible) noexcept nogil
ctypedef SCIP_RETCODE (*ExprGetSymData)(SCIP* scip, SCIP_EXPR* expr, SYM_EXPRDATA** symdata) noexcept nogil


cdef extern from "scip/scip.h" nogil:
    # Expression handlers.
    SCIP_RETCODE SCIPincludeExprhdlr(SCIP* scip, SCIP_EXPRHDLR** exprhdlr, const char* name, const char* desc, unsigned int precedence, ExprEval eval, SCIP_EXPRHDLRDATA* data)
    SCIP_EXPRHDLR* SCIPfindExprhdlr(SCIP* scip, const char* name)
    void SCIPexprhdlrSetCopyFreeHdlr(SCIP_EXPRHDLR* exprhdlr, ExprCopyHdlr copyhdlr, void* freehdlr)
    void SCIPexprhdlrSetCopyFreeData(SCIP_EXPRHDLR* exprhdlr, ExprCopyData copydata, ExprFreeData freedata)
    void SCIPexprhdlrSetCompare(SCIP_EXPRHDLR* exprhdlr, ExprCompare compare)
    void SCIPexprhdlrSetHash(SCIP_EXPRHDLR* exprhdlr, ExprHash hash)
    void SCIPexprhdlrSetIntEval(SCIP_EXPRHDLR* exprhdlr, ExprInteval inteval)
    void SCIPexprhdlrSetEstimate(SCIP_EXPRHDLR* exprhdlr, ExprInitEstimates initestimates, ExprEstimate estimate)
    void SCIPexprhdlrSetDiff(SCIP_EXPRHDLR* exprhdlr, ExprBwdiff bwdiff, ExprFwdiff fwdiff, ExprBwfwdiff bwfwdiff)
    void SCIPexprhdlrSetMonotonicity(SCIP_EXPRHDLR* exprhdlr, ExprMonotonicity monotonicity)
    void SCIPexprhdlrSetGetSymdata(SCIP_EXPRHDLR* exprhdlr, ExprGetSymData getsymdata)
    void SCIPexprhdlrSetReverseProp(SCIP_EXPRHDLR* exprhdlr, ExprReverseProp reverseprop)

    # Expressions.
    SCIP_EXPRDATA* SCIPexprGetData(SCIP_EXPR* expr)
    void SCIPexprSetData(SCIP_EXPR* expr, SCIP_EXPRDATA* exprdata)
    int SCIPexprGetNChildren(SCIP_EXPR* expr)
    SCIP_EXPR** SCIPexprGetChildren(SCIP_EXPR* expr)
    SCIP_Real SCIPexprGetEvalValue(SCIP_EXPR* expr)
    SCIP_Real SCIPexprGetDerivative(SCIP_EXPR* expr)
    SCIP_Real SCIPexprGetDot(SCIP_EXPR* expr)
    SCIP_Real SCIPexprGetBardot(SCIP_EXPR* expr)
    SCIP_INTERVAL SCIPexprGetActivity(SCIP_EXPR* expr)
    SCIP_RETCODE SCIPevalExpr(SCIP* scip, SCIP_EXPR* expr, SCIP_SOL* sol, SCIP_Longint soltag)
    SCIP_RETCODE SCIPevalExprGradient(SCIP* scip, SCIP_EXPR* expr, SCIP_SOL* sol, SCIP_Longint soltag)
    SCIP_RETCODE SCIPevalExprHessianDir(SCIP* scip, SCIP_EXPR* expr, SCIP_SOL* sol, SCIP_Longint soltag, SCIP_SOL* direction)
    SCIP_RETCODE SCIPevalExprActivity(SCIP* scip, SCIP_EXPR* expr)
    int SCIPcompareExpr(SCIP* scip, SCIP_EXPR* expr1, SCIP_EXPR* expr2)
    SCIP_RETCODE SCIPcreateExpr(SCIP* scip, SCIP_EXPR** expr, SCIP_EXPRHDLR* exprhdlr, SCIP_EXPRDATA* exprdata, int nchildren, SCIP_EXPR** children, void* ownercreate, void* ownercreatedata)
    SCIP_RETCODE SCIPcaptureExpr(SCIP_EXPR* expr)
    SCIP_RETCODE SCIPreleaseExpr(SCIP* scip, SCIP_EXPR** expr)
    SCIP_RETCODE SCIPhashExpr(SCIP* scip, SCIP_EXPR* expr, unsigned int* hashval)
    SCIP_RETCODE SCIPcallExprMonotonicity(SCIP* scip, SCIP_EXPR* expr, int childidx, SCIP_MONOTONE* result)
    SCIP_RETCODE SCIPcallExprEstimate(SCIP* scip, SCIP_EXPR* expr, SCIP_INTERVAL* localbounds, SCIP_INTERVAL* globalbounds, SCIP_Real* refpoint, SCIP_Bool overestimate, SCIP_Real targetvalue, SCIP_Real* coefs, SCIP_Real* constant, SCIP_Bool* islocal, SCIP_Bool* success, SCIP_Bool* branchcand)
    SCIP_RETCODE SCIPcallExprInitestimates(SCIP* scip, SCIP_EXPR* expr, SCIP_INTERVAL* bounds, SCIP_Bool overestimate, SCIP_Real** coefs, SCIP_Real* constant, int* nreturned)
    SCIP_RETCODE SCIPcallExprReverseprop(SCIP* scip, SCIP_EXPR* expr, SCIP_INTERVAL bounds, SCIP_INTERVAL* childrenbounds, SCIP_Bool* infeasible)
    SCIP_RETCODE SCIPgetSymDataExpr(SCIP* scip, SCIP_EXPR* expr, SYM_EXPRDATA** symdata)
    SCIP_RETCODE SCIPcallExprInteval(SCIP* scip, SCIP_EXPR* expr, SCIP_INTERVAL* interval, ExprIntevalVar intevalvar, void* intevalvardata)

    # Problem creation.
    SCIP_RETCODE SCIPcreate(SCIP** scip)
    SCIP_RETCODE SCIPfree(SCIP** scip)
    SCIP_RETCODE SCIPcreateProbBasic(SCIP* scip, const char* name)
    SCIP_RETCODE SCIPreadProb(SCIP* scip, const char* filename, const char* extension)
    SCIP_RETCODE SCIPwriteOrigProblem(SCIP* scip, const char* filename, const char* extension, SCIP_Bool genericnames)
    SCIP_RETCODE SCIPcreateVarBasic(SCIP* scip, SCIP_VAR** var, const char* name, SCIP_Real lb, SCIP_Real ub, SCIP_Real obj, SCIP_VARTYPE vartype)
    SCIP_RETCODE SCIPaddVar(SCIP* scip, SCIP_VAR* var)
    SCIP_RETCODE SCIPcaptureVar(SCIP* scip, SCIP_VAR* var)
    SCIP_RETCODE SCIPreleaseVar(SCIP* scip, SCIP_VAR** var)
    SCIP_RETCODE SCIPaddCons(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPreleaseCons(SCIP* scip, SCIP_CONS** cons)
    SCIP_RETCODE SCIPsetObjsense(SCIP* scip, SCIP_OBJSENSE objsense)
    SCIP_OBJSENSE SCIPgetObjsense(SCIP* scip)
    SCIP_RETCODE SCIPaddOrigObjoffset(SCIP* scip, SCIP_Real addval)
    SCIP_VAR** SCIPgetOrigVars(SCIP* scip)
    int SCIPgetNOrigVars(SCIP* scip)
    const char* SCIPvarGetName(SCIP_VAR* var)
    SCIP_Real SCIPinfinity(SCIP* scip)
    SCIP_Bool SCIPisInfinity(SCIP* scip, SCIP_Real val)
    SCIP_Real SCIPfeastol(SCIP* scip)

    # Solving and results.
    SCIP_RETCODE SCIPsolve(SCIP* scip)
    SCIP_RETCODE SCIPfreeTransform(SCIP* scip)
    SCIP_STATUS SCIPgetStatus(SCIP* scip)
    SCIP_Real SCIPgetPrimalbound(SCIP* scip)
    SCIP_Real SCIPgetDualbound(SCIP* scip)
    SCIP_Real SCIPgetGap(SCIP* scip)
    SCIP_Longint SCIPgetNTotalNodes(SCIP* scip)
    SCIP_Real SCIPgetSolvingTime(SCIP* scip)
    int SCIPgetNSols(SCIP* scip)
    SCIP_SOL* SCIPgetBestSol(SCIP* scip)
    SCIP_Real SCIPgetSolVal(SCIP* scip, SCIP_SOL* sol, SCIP_VAR* var)
    SCIP_Real SCIPgetSolOrigObj(SCIP* scip, SCIP_SOL* sol)
    SCIP_RETCODE SCIPcreateOrigSol(SCIP* scip, SCIP_SOL** sol, SCIP_HEUR* heur)
    SCIP_RETCODE SCIPcreateSol(SCIP* scip, SCIP_SOL** sol, SCIP_HEUR* heur)
    SCIP_RETCODE SCIPsetSolVals(SCIP* scip, SCIP_SOL* sol, int nvars, SCIP_VAR** vars, SCIP_Real* vals)
    SCIP_RETCODE SCIPfreeSol(SCIP* scip, SCIP_SOL** sol)
    SCIP_RETCODE SCIPcheckSolOrig(SCIP* scip, SCIP_SOL* sol, SCIP_Bool* feasible, SCIP_Bool printreason, SCIP_Bool completely)

    # Parameters and messages.
    SCIP_RETCODE SCIPsetEmphasis(SCIP* scip, SCIP_PARAMEMPHASIS paramemphasis, SCIP_Bool quiet)
    SCIP_RETCODE SCIPsetBoolParam(SCIP* scip, const char* name, SCIP_Bool value)
    SCIP_RETCODE SCIPsetIntParam(SCIP* scip, const char* name, int value)
    SCIP_RETCODE SCIPsetLongintParam(SCIP* scip, const char* name, SCIP_Longint value)
    SCIP_RETCODE SCIPsetRealParam(SCIP* scip, const char* name, SCIP_Real value)
    SCIP_RETCODE SCIPsetCharParam(SCIP* scip, const char* name, char value)
    SCIP_RETCODE SCIPsetStringParam(SCIP* scip, const char* name, const char* value)
    SCIP_RETCODE SCIPgetBoolParam(SCIP* scip, const char* name, SCIP_Bool* value)
    SCIP_RETCODE SCIPgetIntParam(SCIP* scip, const char* name, int* value)
    SCIP_RETCODE SCIPgetLongintParam(SCIP* scip, const char* name, SCIP_Longint* value)
    SCIP_RETCODE SCIPgetRealParam(SCIP* scip, const char* name, SCIP_Real* value)
    SCIP_RETCODE SCIPgetCharParam(SCIP* scip, const char* name, char* value)
    SCIP_RETCODE SCIPgetStringParam(SCIP* scip, const char* name, char** value)
    SCIP_PARAM** SCIPgetParams(SCIP* scip)
    int SCIPgetNParams(SCIP* scip)
    SCIP_PARAM* SCIPgetParam(SCIP* scip, const char* name)
    SCIP_PARAMTYPE SCIPparamGetType(SCIP_PARAM* param)
    const char* SCIPparamGetName(SCIP_PARAM* param)
    const char* SCIPparamGetDesc(SCIP_PARAM* param)
    void SCIPsetMessagehdlrQuiet(SCIP* scip, SCIP_Bool quiet)
    int SCIPmajorVersion()
    int SCIPminorVersion()
    int SCIPtechVersion()

    # SCIP's block-memory macros take a pointer to the pointer they set, so each element type
    # gets its own declaration with the same C name.
    SCIP_RETCODE allocSymData "SCIPallocBlockMemory"(SCIP* scip, SYM_EXPRDATA** ptr)
    SCIP_RETCODE allocRealArray "SCIPallocBlockMemoryArray"(SCIP* scip, SCIP_Real** ptr, int num)
    SCIP_RETCODE allocExprArray "SCIPallocBlockMemoryArray"(SCIP* scip, SCIP_EXPR*** ptr, int num)
    void freeSymData "SCIPfreeBlockMemory"(SCIP* scip, SYM_EXPRDATA** ptr)
    void freeRealArray "SCIPfreeBlockMemoryArray"(SCIP* scip, SCIP_Real** ptr, int num)
    void freeExprArray "SCIPfreeBlockMemoryArray"(SCIP* scip, SCIP_EXPR*** ptr, int num)


cdef extern from "scip/scipdefplugins.h" nogil:
    SCIP_RETCODE SCIPincludeDefaultPlugins(SCIP* scip)


cdef extern from "scip/expr_var.h" nogil:
    SCIP_RETCODE SCIPcreateExprVar(SCIP* scip, SCIP_EXPR** expr, SCIP_VAR* var, void* ownercreate, void* ownercreatedata)


cdef extern from "scip/expr_sum.h" nogil:
    SCIP_RETCODE SCIPcreateExprSum(SCIP* scip, SCIP_EXPR** expr, int nchildren, SCIP_EXPR** children, SCIP_Real* coefficients, SCIP_Real constant, void* ownercreate, void* ownercreatedata)


cdef extern from "scip/cons_linear.h" nogil:
    SCIP_RETCODE SCIPcreateConsBasicLinear(SCIP* scip, SCIP_CONS** cons, const char* name, int nvars, SCIP_VAR** vars, SCIP_Real* vals, SCIP_Real lhs, SCIP_Real rhs)


cdef extern from "scip/cons_nonlinear.h" nogil:
    SCIP_RETCODE SCIPcreateConsBasicNonlinear(SCIP* scip, SCIP_CONS** cons, const char* name, SCIP_EXPR* expr, SCIP_Real lhs, SCIP_Real rhs)


cdef extern from "symmetry/struct_symmetry.h":
    pass
