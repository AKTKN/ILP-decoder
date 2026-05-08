"""ILP formulation builders for the decoder.

This module builds a Gurobi model from a parity check matrix and prior.
Concrete solver calls are intentionally isolated from higher-level orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .models import DecoderConfig, Prior

if TYPE_CHECKING:
    from gurobipy import Constr, Env, LinExpr, Model, Var
else:  # pragma: no cover - runtime fallback without gurobipy
    Constr = Env = LinExpr = Model = Var = Any

ParityCheckMatrix: TypeAlias = NDArray[np.int64]
Observables: TypeAlias = NDArray[np.int64]
Weights: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class BuildModelResult:
	"""Container for objects produced during model construction.

	Attributes:
		model: Initialized Gurobi model instance.
		error_vars: Primary error indicator variables.
		auxiliary_vars: Named groups of auxiliary variables.
		syndrome_constraints: Constraints tying errors to the syndrome.
		objective: Optional objective expression if constructed.
	"""

	model: Model
	error_vars: Sequence[Var]
	auxiliary_vars: Mapping[str, Sequence[Var]]
	syndrome_constraints: Sequence[Constr]
	objective: Optional[LinExpr] = None


def _apply_config_params(model: "Model", config: DecoderConfig) -> None:
	"""Apply standard solver parameters from DecoderConfig."""

	if config.time_limit_s is not None:
		model.setParam("TimeLimit", config.time_limit_s)
	if config.mip_gap is not None:
		model.setParam("MIPGap", config.mip_gap)
	if config.threads is not None:
		model.setParam("Threads", config.threads)
	if config.random_seed is not None:
		model.setParam("Seed", config.random_seed)
	model.setParam("OutputFlag", 1 if config.log_to_console else 0)
	if config.enable_lazy_constraints:
		model.setParam("LazyConstraints", 1)

	# Allow opt-in for solver-specific settings.
	for key, value in config.solver_options.items():
		model.setParam(key, value)


def _compute_weights_from_prior(prior: Prior) -> Weights:
	"""Convert prior probabilities into linear weights.

	Uses log-odds by default. Adjust this function if you want a different
	objective definition.
	"""

	eps = 1e-12
	clipped = np.clip(prior, eps, 1.0 - eps)
	return np.log((1.0 - clipped) / clipped)


def _iter_row_indices(parity_check_matrix: ParityCheckMatrix) -> Sequence[Sequence[int]]:
	"""Iterate column indices for each parity-check row.

	Supports dense numpy arrays and scipy sparse matrices.
	"""

	try:
		from scipy import sparse
	except ImportError:  # pragma: no cover - optional dependency
		sparse = None

	if sparse is not None and sparse.issparse(parity_check_matrix):
		pcm_csr = parity_check_matrix.tocsr()
		row_indices = []
		for i in range(pcm_csr.shape[0]):
			start, end = pcm_csr.indptr[i], pcm_csr.indptr[i + 1]
			row_indices.append(list(pcm_csr.indices[start:end]))
		return row_indices

	pcm_dense = np.asarray(parity_check_matrix, dtype=int)
	return [list(np.nonzero(row)[0]) for row in pcm_dense]


def build_base_model(
	env: Env,
	parity_check_matrix: ParityCheckMatrix,
	observables: Observables,
	config: DecoderConfig,
	*,
	prior: Optional[Prior] = None,
	weight_vector: Optional[Weights] = None,
	extra_constraints: Optional[Mapping[str, Any]] = None,
) -> BuildModelResult:
	"""Create a base optimization model for ILP decoding.

	Args:
		env: Gurobi environment object used to create the model.
		parity_check_matrix: Parity check matrix defining stabilizer constraints.
		observables: Logical observable definitions for logical error tracking.
		config: Decoder configuration and solver options.
		prior: Optional prior probability for each error variable.
		weight_vector: Optional weight for each error variable.
		extra_constraints: Optional named constraints for extensibility.

	Returns:
		A structured bundle containing the model, variables, and constraints.
	"""

	import gurobipy as gp

	# Preserve sparse matrices to avoid dense materialization.
	num_checks, num_errors = parity_check_matrix.shape
	if observables.shape[1] != num_errors:
		raise ValueError("Observables must have the same column size as the check matrix.")

	if weight_vector is None:
		if prior is not None:
			weight_vector = _compute_weights_from_prior(np.asarray(prior, dtype=float))
		else:
			# Default to unit weights when no prior is provided.
			weight_vector = np.ones(num_errors, dtype=float)
	else:
		weight_vector = np.asarray(weight_vector, dtype=float)

	if weight_vector.shape[0] != num_errors:
		raise ValueError("Weight vector length must match number of error variables.")

	# Build the base model with all structural constraints but no syndrome RHS set.
	model = gp.Model(env=env)
	_apply_config_params(model, config)

	# Primary error indicator variables (binary).
	error_vars = model.addVars(num_errors, vtype=gp.GRB.BINARY, name="e")

	# Integer slack variables enforce parity constraints: H * e - 2 * t = syndrome.
	# Each row creates one integer variable t to encode parity in linear form.
	parity_slack = model.addVars(num_checks, vtype=gp.GRB.INTEGER, lb=0, name="t")

	syndrome_constraints = []
	row_indices = _iter_row_indices(parity_check_matrix)
	for row, cols in enumerate(row_indices):
		expr = gp.LinExpr()
		for col in cols:
			expr.add(error_vars[col])

		# RHS is set later based on the syndrome for each shot.
		constr = model.addConstr(expr - 2 * parity_slack[row] == 0, name=f"syn_{row}")
		syndrome_constraints.append(constr)

	# Objective: minimize weighted sum of error indicators.
	objective = gp.quicksum(weight_vector[j] * error_vars[j] for j in range(num_errors))
	model.setObjective(objective, gp.GRB.MINIMIZE)

	# Optional extensibility hook: allow caller to add custom constraints.
	if extra_constraints:
		custom_builder = extra_constraints.get("builder")
		if callable(custom_builder):
			custom_builder(model, error_vars, parity_slack)

	model.update()

	return BuildModelResult(
		model=model,
		error_vars=[error_vars[j] for j in range(num_errors)],
		auxiliary_vars={"parity_slack": [parity_slack[i] for i in range(num_checks)]},
		syndrome_constraints=syndrome_constraints,
		objective=objective,
	)
