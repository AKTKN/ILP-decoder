"""CPLEX backend components for the ILP decoder.

This module provides the environment holder, variable/constraint proxies, and
solver backend for the CPLEX optimizer.  The proxy classes (CplexVar,
CplexConstr) expose the same attribute interface as their gurobipy counterparts
so that core.py can remain solver-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .models import DecodeResult, DecoderConfig


@dataclass
class CplexEnv:
    """Environment holder for the CPLEX backend.

    Pass an instance to DecoderDependencies(env=...) when using the CPLEX
    backend.  Reserved for future global CPLEX configuration (e.g., licence
    path, default parameter overrides).
    """

    pass


class CplexVar:
    """Proxy for a CPLEX variable that exposes .X to retrieve the solution value.

    Mirrors the gurobipy.Var.X interface so that core.py does not need to
    branch on the solver type when reading solution values.
    """

    __slots__ = ("_model", "_index")

    def __init__(self, model: Any, index: int) -> None:
        self._model = model
        self._index = index

    @property
    def X(self) -> float:
        return float(self._model.solution.get_values(self._index))


class CplexConstr:
    """Proxy for a CPLEX linear constraint that exposes a settable .RHS.

    Mirrors the gurobipy.Constr.RHS interface so that core.py and test code
    can update syndrome right-hand sides without knowing the solver type.
    """

    __slots__ = ("_model", "_index")

    def __init__(self, model: Any, index: int) -> None:
        self._model = model
        self._index = index

    @property
    def RHS(self) -> float:
        return float(self._model.linear_constraints.get_rhs(self._index))

    @RHS.setter
    def RHS(self, value: float) -> None:
        self._model.linear_constraints.set_rhs(self._index, float(value))


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

_CPLEX_STATUS_LABELS: Mapping[int, str] = {
    1: "OPTIMAL",    # LP optimal
    5: "OPTIMAL",    # LP optimal (obj tolerance)
    101: "OPTIMAL",  # MIP optimal
    102: "OPTIMAL",  # MIP within tolerance
    103: "INFEASIBLE",
    107: "TIME_LIMIT",  # time limit, integer feasible
    108: "TIME_LIMIT",  # time limit, no integer feasible
    109: "TIME_LIMIT",  # stopped, integer feasible
    110: "TIME_LIMIT",  # stopped, no integer feasible
}

# Status codes that indicate at least one primal solution is available.
_CPLEX_HAS_SOLUTION: frozenset[int] = frozenset([1, 5, 101, 102, 107, 109, 111, 113])


def _cplex_status_label(status_code: int) -> str:
    return _CPLEX_STATUS_LABELS.get(status_code, str(status_code))


# ---------------------------------------------------------------------------
# Config application
# ---------------------------------------------------------------------------

def _apply_cplex_config_params(prob: Any, config: DecoderConfig) -> None:
    """Apply DecoderConfig settings to a cplex.Cplex instance."""

    if config.log_to_console:
        # Restore default streams (no-op if already set).
        import sys
        prob.set_log_stream(sys.stdout)
        prob.set_results_stream(sys.stdout)
        prob.set_warning_stream(sys.stdout)
        prob.set_error_stream(sys.stderr)
    else:
        prob.set_log_stream(None)
        prob.set_results_stream(None)
        prob.set_warning_stream(None)
        prob.set_error_stream(None)

    if config.time_limit_s is not None:
        prob.parameters.timelimit.set(float(config.time_limit_s))
    if config.mip_gap is not None:
        prob.parameters.mip.tolerances.mipgap.set(float(config.mip_gap))
    if config.threads is not None:
        prob.parameters.threads.set(int(config.threads))
    if config.random_seed is not None:
        prob.parameters.randomseed.set(int(config.random_seed))

    # solver_options is intentionally not forwarded to CPLEX because CPLEX
    # parameters use a hierarchical object API rather than string keys.


# ---------------------------------------------------------------------------
# Solver backend
# ---------------------------------------------------------------------------

class CplexBackend:
    """Solver backend that drives a cplex.Cplex model.

    Implements the SolverBackend protocol used by ILPDecoder, mirroring
    GurobiBackend but using the CPLEX Python API.
    """

    def solve(
        self,
        model_result: Any,
        syndrome: Any,
        *,
        config: DecoderConfig,
    ) -> DecodeResult:
        prob = model_result.model  # cplex.Cplex instance

        _apply_cplex_config_params(prob, config)

        syndrome = np.asarray(syndrome, dtype=int)
        if syndrome.shape[0] != len(model_result.syndrome_constraints):
            raise ValueError("Syndrome length must match number of parity checks.")

        # Update parity constraint RHS values for this shot.
        for i, constr in enumerate(model_result.syndrome_constraints):
            constr.RHS = int(syndrome[i])  # uses CplexConstr proxy

        t0 = time.perf_counter()
        prob.solve()
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        status_code = prob.solution.get_status()
        status = _cplex_status_label(status_code)

        if status_code in _CPLEX_HAS_SOLUTION:
            error_vector = np.array(
                [int(round(var.X)) for var in model_result.error_vars], dtype=np.int64
            )
            objective_value = float(prob.solution.get_objective_value())
            success = True
        else:
            error_vector = np.zeros(len(model_result.error_vars), dtype=np.int64)
            objective_value = None
            success = False

        return DecodeResult(
            success=success,
            error_vector=error_vector,
            objective_value=objective_value,
            runtime_ms=runtime_ms,
            status=status,
            predicted_observables=None,
            metadata={"status_code": status_code, "solver": "cplex"},
        )
