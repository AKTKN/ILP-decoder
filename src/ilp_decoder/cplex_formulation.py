"""CPLEX ILP formulation builders for the decoder.

This module mirrors formulation.py but targets the CPLEX Python API.
The public functions return BuildModelResult objects whose model, error_vars,
auxiliary_vars, and syndrome_constraints fields hold CPLEX-compatible proxy
objects (CplexVar, CplexConstr) rather than gurobipy objects.

Decoding logic (weights, constraint structure, logical-gap formulation) is
identical to the Gurobi implementation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .cplex_backend import (
    CplexConstr,
    CplexEnv,
    CplexVar,
    _apply_cplex_config_params,
)
from .formulation import (
    BuildModelResult,
    Observables,
    ParityCheckMatrix,
    Weights,
    _compute_weights_from_prior,
    _iter_row_indices,
)
from .models import DecoderConfig, Prior


def build_base_model_cplex(
    env: CplexEnv,
    parity_check_matrix: ParityCheckMatrix,
    observables: Observables,
    config: DecoderConfig,
    *,
    prior: Optional[Prior] = None,
    weight_vector: Optional[Weights] = None,
    extra_constraints: Optional[Mapping[str, Any]] = None,
) -> BuildModelResult:
    """Create a CPLEX MIP model for ILP decoding.

    The model structure is identical to build_base_model:
        minimise  sum_j  w[j] * e[j]
        subject to
            H * e - 2 * t  =  syndrome   (set per shot via CplexConstr.RHS)
            e[j] in {0, 1},  t[i] in Z_{>=0}

    Args:
        env: CplexEnv instance (no Gurobi Env needed for CPLEX).
        parity_check_matrix: Binary parity check matrix (m x n).
        observables: Logical observable matrix (k x n).
        config: Decoder configuration and solver options.
        prior: Optional prior probability for each error variable.
        weight_vector: Optional pre-computed weights (overrides prior).
        extra_constraints: Optional dict with a "builder" callable for
            custom constraints; receives (prob, error_var_proxies,
            parity_slack_proxies).

    Returns:
        BuildModelResult with CPLEX proxy objects.
    """

    import cplex

    num_checks, num_errors = parity_check_matrix.shape
    if observables.shape[1] != num_errors:
        raise ValueError("Observables must have the same column size as the check matrix.")

    if weight_vector is None:
        if prior is not None:
            weight_vector = _compute_weights_from_prior(np.asarray(prior, dtype=float))
        else:
            weight_vector = np.ones(num_errors, dtype=float)
    else:
        weight_vector = np.asarray(weight_vector, dtype=float)

    if weight_vector.shape[0] != num_errors:
        raise ValueError("Weight vector length must match number of error variables.")

    prob = cplex.Cplex()
    _apply_cplex_config_params(prob, config)
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Binary error variables e_0 ... e_{n-1}  (indices 0 .. num_errors-1)
    prob.variables.add(
        names=[f"e_{j}" for j in range(num_errors)],
        types=["B"] * num_errors,
        obj=list(weight_vector.astype(float)),
    )

    # Integer slack variables t_0 ... t_{m-1}  (indices num_errors .. num_errors+num_checks-1)
    t_start = num_errors
    prob.variables.add(
        names=[f"t_{i}" for i in range(num_checks)],
        types=["I"] * num_checks,
        lb=[0.0] * num_checks,
    )

    # Parity constraints: sum_{col in row} e[col] - 2*t[row] = syndrome[row]
    # RHS is initialised to 0 and updated per shot via CplexConstr.RHS.
    row_indices = _iter_row_indices(parity_check_matrix)
    syndrome_constraints = []
    for row, cols in enumerate(row_indices):
        ind = [int(c) for c in cols] + [t_start + row]
        val = [1.0] * len(cols) + [-2.0]
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=ind, val=val)],
            senses=["E"],
            rhs=[0.0],
            names=[f"syn_{row}"],
        )
        # Constraint index equals the loop counter because constraints are
        # appended in order starting from 0.
        syndrome_constraints.append(CplexConstr(prob, row))

    if extra_constraints:
        custom_builder = extra_constraints.get("builder")
        if callable(custom_builder):
            error_var_proxies = [CplexVar(prob, j) for j in range(num_errors)]
            parity_slack_proxies = [CplexVar(prob, t_start + i) for i in range(num_checks)]
            custom_builder(prob, error_var_proxies, parity_slack_proxies)

    error_vars = [CplexVar(prob, j) for j in range(num_errors)]
    parity_slack_vars = [CplexVar(prob, t_start + i) for i in range(num_checks)]

    return BuildModelResult(
        model=prob,
        error_vars=error_vars,
        auxiliary_vars={"parity_slack": parity_slack_vars},
        syndrome_constraints=syndrome_constraints,
    )


def build_logical_gap_model_cplex(
    env: CplexEnv,
    parity_check_matrix: ParityCheckMatrix,
    observables: Observables,
    config: DecoderConfig,
    *,
    logical_class: Sequence[int],
    prior: Optional[Prior] = None,
    weight_vector: Optional[Weights] = None,
    extra_constraints: Optional[Mapping[str, Any]] = None,
) -> BuildModelResult:
    """Create a CPLEX stage-2 model that excludes a target logical class.

    Extends build_base_model_cplex with additional variables and constraints
    for logical gap computation.  The formulation is identical to
    build_logical_gap_model:

        For each observable row i:
            L[i] @ e + logical_class[i] - 2*u[i] = z[i]
        sum_i z[i] >= 1

    Variable layout:
        0          .. num_errors-1          : e (binary error indicators)
        num_errors .. num_errors+m-1        : t (parity slack, integer >=0)
        num_errors+m .. num_errors+m+k-1   : z (binary observable XOR)
        num_errors+m+k .. num_errors+m+2k-1: u (logical slack, integer >=0)

    Args:
        env: CplexEnv instance.
        logical_class: Current logical class prediction (length k).
        (remaining args identical to build_base_model_cplex)

    Returns:
        BuildModelResult with auxiliary_vars["logical_xor"] holding CplexVar
        proxies for z variables.
    """

    import cplex

    num_checks, num_errors = parity_check_matrix.shape
    if observables.shape[1] != num_errors:
        raise ValueError("Observables must have the same column size as the check matrix.")

    if weight_vector is None:
        if prior is not None:
            weight_vector = _compute_weights_from_prior(np.asarray(prior, dtype=float))
        else:
            weight_vector = np.ones(num_errors, dtype=float)
    else:
        weight_vector = np.asarray(weight_vector, dtype=float)

    if weight_vector.shape[0] != num_errors:
        raise ValueError("Weight vector length must match number of error variables.")

    logical_class = np.asarray(logical_class, dtype=int).ravel() % 2
    num_logicals = observables.shape[0]
    if logical_class.shape[0] != num_logicals:
        raise ValueError("Logical class length must match number of observables.")

    prob = cplex.Cplex()
    _apply_cplex_config_params(prob, config)
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Error variables  (indices 0 .. n-1)
    prob.variables.add(
        names=[f"e_{j}" for j in range(num_errors)],
        types=["B"] * num_errors,
        obj=list(weight_vector.astype(float)),
    )

    # Parity slack  (indices n .. n+m-1)
    t_start = num_errors
    prob.variables.add(
        names=[f"t_{i}" for i in range(num_checks)],
        types=["I"] * num_checks,
        lb=[0.0] * num_checks,
    )

    # Z variables – binary XOR indicators  (indices n+m .. n+m+k-1)
    z_start = t_start + num_checks
    prob.variables.add(
        names=[f"z_{i}" for i in range(num_logicals)],
        types=["B"] * num_logicals,
    )

    # U variables – integer slack for logical constraints  (n+m+k .. n+m+2k-1)
    u_start = z_start + num_logicals
    prob.variables.add(
        names=[f"u_{i}" for i in range(num_logicals)],
        types=["I"] * num_logicals,
        lb=[0.0] * num_logicals,
    )

    # Parity constraints (RHS updated per shot)
    row_indices = _iter_row_indices(parity_check_matrix)
    syndrome_constraints = []
    for row, cols in enumerate(row_indices):
        ind = [int(c) for c in cols] + [t_start + row]
        val = [1.0] * len(cols) + [-2.0]
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=ind, val=val)],
            senses=["E"],
            rhs=[0.0],
            names=[f"syn_{row}"],
        )
        syndrome_constraints.append(CplexConstr(prob, row))

    # Logical constraints:
    #   sum_{col} e[col]  -  2*u[row]  -  z[row]  =  -logical_class[row]
    # (equivalent to: sum e + logical_class - 2u = z)
    logical_row_indices = _iter_row_indices(observables)
    num_parity_constrs = num_checks
    for row, cols in enumerate(logical_row_indices):
        ind = [int(c) for c in cols] + [u_start + row, z_start + row]
        val = [1.0] * len(cols) + [-2.0, -1.0]
        rhs = -float(logical_class[row])
        constr_idx = num_parity_constrs + row
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=ind, val=val)],
            senses=["E"],
            rhs=[rhs],
            names=[f"logical_{row}"],
        )

    # At least one observable must flip: sum(z) >= 1
    if num_logicals > 0:
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[z_start + i for i in range(num_logicals)],
                val=[1.0] * num_logicals,
            )],
            senses=["G"],
            rhs=[1.0],
            names=["logical_diff"],
        )

    if extra_constraints:
        custom_builder = extra_constraints.get("builder")
        if callable(custom_builder):
            error_var_proxies = [CplexVar(prob, j) for j in range(num_errors)]
            parity_slack_proxies = [CplexVar(prob, t_start + i) for i in range(num_checks)]
            custom_builder(prob, error_var_proxies, parity_slack_proxies)

    error_vars = [CplexVar(prob, j) for j in range(num_errors)]
    parity_slack_vars = [CplexVar(prob, t_start + i) for i in range(num_checks)]
    z_vars = [CplexVar(prob, z_start + i) for i in range(num_logicals)]
    u_vars = [CplexVar(prob, u_start + i) for i in range(num_logicals)]

    return BuildModelResult(
        model=prob,
        error_vars=error_vars,
        auxiliary_vars={
            "parity_slack": parity_slack_vars,
            "logical_xor": z_vars,
            "logical_slack": u_vars,
        },
        syndrome_constraints=syndrome_constraints,
    )
