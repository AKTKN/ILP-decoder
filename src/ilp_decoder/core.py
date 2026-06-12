"""Public decoder interface and dependency-injected orchestration.

This module wires together model construction and solver execution.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence, TYPE_CHECKING

import numpy as np

from .formulation import (
    BuildModelResult,
    Observables,
    ParityCheckMatrix,
    Weights,
    build_base_model,
    build_logical_gap_model,
)
from .models import DecodeResult, DecoderConfig, Prior, Syndrome

if TYPE_CHECKING:
    from gurobipy import Env


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------

class OptimizerBackend(Enum):
    """Supported optimizer backends."""

    GUROBI = "gurobi"
    CPLEX = "cplex"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class ModelBuilder(Protocol):
    """Callable interface for constructing solver models."""

    def __call__(
        self,
        env: Any,
        parity_check_matrix: ParityCheckMatrix,
        observables: Observables,
        config: DecoderConfig,
        *,
        prior: Optional[Prior] = None,
        weight_vector: Optional[Weights] = None,
        extra_constraints: Optional[Mapping[str, Any]] = None,
    ) -> BuildModelResult:
        ...


class GapModelBuilder(Protocol):
    """Callable interface for constructing logical-gap solver models."""

    def __call__(
        self,
        env: Any,
        parity_check_matrix: ParityCheckMatrix,
        observables: Observables,
        config: DecoderConfig,
        *,
        logical_class: Sequence[int],
        prior: Optional[Prior] = None,
        weight_vector: Optional[Weights] = None,
        extra_constraints: Optional[Mapping[str, Any]] = None,
    ) -> BuildModelResult:
        ...


class SolverBackend(Protocol):
    """Interface for solver execution and result extraction."""

    def solve(
        self,
        model_result: BuildModelResult,
        syndrome: Syndrome,
        *,
        config: DecoderConfig,
    ) -> DecodeResult:
        ...


# ---------------------------------------------------------------------------
# Gurobi helpers (kept in this module to avoid circular imports)
# ---------------------------------------------------------------------------

def _apply_config_params(model: Any, config: DecoderConfig) -> None:
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


def _status_label(status_code: int) -> str:
    """Return a human-readable status string for common Gurobi codes."""

    try:
        import gurobipy as gp
    except ImportError:  # pragma: no cover - optional dependency
        return str(status_code)

    labels = {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
    }
    return labels.get(status_code, str(status_code))


def _compute_observable_predictions(
    observables: Observables, error_vector: np.ndarray
) -> np.ndarray:
    """Project a physical error vector onto logical observables."""

    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover - optional dependency
        sparse = None

    if sparse is not None and sparse.issparse(observables):
        return (observables @ error_vector) % 2

    obs_dense = np.asarray(observables, dtype=int)
    return (obs_dense @ error_vector) % 2


def _release_solver_model(model: Any) -> None:
    """Release native solver resources held by a model, if supported."""

    for method_name in ("end", "dispose", "close"):
        method = getattr(model, method_name, None)
        if callable(method):
            method()
            return


def _resolve_logical_gap_flag(
    legacy_flag: bool,
    explicit_flag: Optional[bool],
) -> bool:
    """Resolve `get_logicalgap` and `get_logical_gap` into one boolean."""

    if explicit_flag is None:
        return legacy_flag
    if legacy_flag and not explicit_flag:
        raise ValueError(
            "Received conflicting logical-gap flags: "
            "get_logicalgap=True and get_logical_gap=False."
        )
    return explicit_flag


# ---------------------------------------------------------------------------
# Gurobi backend
# ---------------------------------------------------------------------------

class GurobiBackend:
    """Default solver backend using Gurobi."""

    def solve(
        self,
        model_result: BuildModelResult,
        syndrome: Syndrome,
        *,
        config: DecoderConfig,
    ) -> DecodeResult:
        import gurobipy as gp

        model = model_result.model
        _apply_config_params(model, config)

        # Clear any previous solution state before reusing the model.
        model.reset()

        syndrome = np.asarray(syndrome, dtype=int)
        if syndrome.shape[0] != len(model_result.syndrome_constraints):
            raise ValueError("Syndrome length must match number of parity checks.")

        # Update parity constraints with the current syndrome values.
        # This keeps the model structure fixed while swapping RHS per shot.
        for i, constr in enumerate(model_result.syndrome_constraints):
            constr.RHS = int(syndrome[i])

        model.update()
        model.optimize()

        status_code = int(model.Status)
        status = _status_label(status_code)
        runtime_ms = float(model.Runtime) * 1000.0

        # Extract a binary solution if a feasible solution exists.
        if model.SolCount > 0:
            error_vector = np.array(
                [int(round(var.X)) for var in model_result.error_vars], dtype=np.int64
            )
            objective_value = float(model.ObjVal)
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
            metadata={"status_code": status_code, "solver": "gurobi"},
        )


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def _validate_env_backend(env: Any, backend: OptimizerBackend) -> None:
    """Raise TypeError if env and backend are inconsistent."""

    from .cplex_backend import CplexEnv

    if backend == OptimizerBackend.CPLEX:
        if not isinstance(env, CplexEnv):
            raise TypeError(
                f"CPLEX backend requires a CplexEnv instance, got {type(env).__name__}. "
                "Create one with: from ilp_decoder.cplex_backend import CplexEnv; env = CplexEnv()"
            )
    elif backend == OptimizerBackend.GUROBI:
        if isinstance(env, CplexEnv):
            raise TypeError(
                "GUROBI backend does not accept a CplexEnv. "
                "Pass a gurobipy.Env instance instead."
            )


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecoderDependencies:
    """Injected dependencies for the decoder.

    Attributes:
        env: Solver environment instance.  For Gurobi, pass a gurobipy.Env.
             For CPLEX, pass a CplexEnv from ilp_decoder.cplex_backend.
        backend: Which optimizer to use (GUROBI or CPLEX).  The decoder
                 validates that env matches the selected backend and raises
                 TypeError if they are inconsistent.
        model_builder: Function that builds the base optimization model.
                       Defaults to the appropriate builder for the chosen backend.
        gap_model_builder: Function that builds the logical-gap model.
                           Defaults to the appropriate builder for the chosen backend.
        solver_backend: Strategy that solves the model and extracts results.
                        Defaults to the appropriate backend for the chosen backend.
    """

    env: Any
    backend: OptimizerBackend = OptimizerBackend.GUROBI
    model_builder: Optional[ModelBuilder] = None
    gap_model_builder: Optional[GapModelBuilder] = None
    solver_backend: Optional[SolverBackend] = None


# ---------------------------------------------------------------------------
# High-level decoder
# ---------------------------------------------------------------------------

class ILPDecoder:
    """High-level decoder interface.

    This class coordinates model construction and solver execution but does not
    implement the mathematical or solver-specific logic.
    """

    def __init__(
        self,
        *,
        parity_check_matrix: ParityCheckMatrix,
        observables: Observables,
        prior: Prior,
        config: DecoderConfig,
        deps: DecoderDependencies,
    ) -> None:
        """Initialize a decoder with configuration and dependencies.

        Args:
            parity_check_matrix: Parity check matrix defining stabilizers.
            observables: Logical observables used for predictions.
            prior: Prior probability for each error variable.
            config: Default decoder configuration.
            deps: Injected dependencies for model building and solving.
        """

        # Validate that env type matches the declared backend.
        _validate_env_backend(deps.env, deps.backend)

        num_errors = parity_check_matrix.shape[1]
        if observables.shape[1] != num_errors:
            raise ValueError("Observables must have the same column size as the check matrix.")

        prior = np.asarray(prior, dtype=float)
        if prior.shape[0] != num_errors:
            raise ValueError("Prior length must match number of error variables.")

        self._parity_check_matrix = parity_check_matrix
        self._observables = observables
        self._prior = prior
        self._config = config
        self._deps = deps
        self._closed = False

        # Select builder and backend defaults based on the declared backend.
        if deps.backend == OptimizerBackend.CPLEX:
            from .cplex_formulation import (
                build_base_model_cplex,
                build_logical_gap_model_cplex,
            )
            from .cplex_backend import CplexBackend
            self._model_builder: ModelBuilder = deps.model_builder or build_base_model_cplex
            self._gap_model_builder: GapModelBuilder = deps.gap_model_builder or build_logical_gap_model_cplex
            self._solver_backend: SolverBackend = deps.solver_backend or CplexBackend()
        else:
            self._model_builder = deps.model_builder or build_base_model
            self._gap_model_builder = deps.gap_model_builder or build_logical_gap_model
            self._solver_backend = deps.solver_backend or GurobiBackend()

        # Build a reusable base model upfront for fast decoding.
        self._base_model_result = self._model_builder(
            deps.env,
            parity_check_matrix,
            observables,
            config,
            prior=prior,
        )

    def close(self) -> None:
        """Release native resources held by the cached base model."""

        if self._closed:
            return
        self._closed = True
        _release_solver_model(self._base_model_result.model)

    def __enter__(self) -> "ILPDecoder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ILPDecoder has been closed.")

    @property
    def config(self) -> DecoderConfig:
        """Return the default decoder configuration."""

        return self._config

    def decode(
        self,
        syndrome: Syndrome,
        *,
        weight_vector: Optional[Weights] = None,
        extra_constraints: Optional[Mapping[str, Any]] = None,
        config: Optional[DecoderConfig] = None,
    ) -> np.ndarray:
        """Decode a single syndrome sample and return predicted observables.

        Args:
            syndrome: Syndrome vector for the current shot.
            weight_vector: Optional per-variable weights.
            extra_constraints: Optional named constraint groups.
            config: Optional override for decoder configuration.

        Returns:
            Binary numpy array of predicted observable flips.

        """
        result = self.decode_result(
            syndrome,
            weight_vector=weight_vector,
            extra_constraints=extra_constraints,
            config=config,
        )
        return result.predicted_observables

    def decode_result(
        self,
        syndrome: Syndrome,
        *,
        weight_vector: Optional[Weights] = None,
        extra_constraints: Optional[Mapping[str, Any]] = None,
        config: Optional[DecoderConfig] = None,
        get_logicalgap: bool = False,
        get_logical_gap: Optional[bool] = None,
        get_gap_detail: bool = False,
        logical_gap_flip_last_detector: bool = False,
    ) -> DecodeResult:
        """Decode a single syndrome sample and return a full DecodeResult.

        This is useful for debugging and metric computation.
        Set get_logicalgap=True to compute the logical gap and store it in metadata.
        """

        self._ensure_open()
        get_logicalgap = _resolve_logical_gap_flag(get_logicalgap, get_logical_gap)
        if get_gap_detail and not get_logicalgap:
            raise ValueError("get_gap_detail requires get_logicalgap=True.")

        active_config = config or self._config

        # Reuse the cached model unless the caller supplies overrides.
        if weight_vector is None and extra_constraints is None and config is None:
            model_result = self._base_model_result
            owns_model_result = False
        else:
            # Rebuild when overrides are supplied to keep the base model reusable.
            model_result = self._model_builder(
                self._deps.env,
                self._parity_check_matrix,
                self._observables,
                active_config,
                prior=self._prior,
                weight_vector=weight_vector,
                extra_constraints=extra_constraints,
            )
            owns_model_result = True

        try:
            # Solve the ILP and map the solution into observable predictions.
            backend_result = self._solver_backend.solve(
                model_result,
                syndrome,
                config=active_config,
            )
            predicted_observables = _compute_observable_predictions(
                self._observables, backend_result.error_vector
            )

            metadata = dict(backend_result.metadata)
            if get_logicalgap:
                gap_metadata = self._compute_logical_gap(
                    syndrome,
                    predicted_observables,
                    backend_result.objective_value,
                    weight_vector=weight_vector,
                    extra_constraints=extra_constraints,
                    config=active_config,
                    flip_last_detector=logical_gap_flip_last_detector,
                )
                metadata["logical_gap"] = gap_metadata["logical_gap"]
                metadata["obs_flip_idx"] = gap_metadata["obs_flip_idx"]
                if get_gap_detail:
                    metadata["gap_detail"] = {
                        "stage1_weight": float(backend_result.objective_value),
                        "stage2_weight": gap_metadata["stage2_weight"],
                        "stage2_error_vector": gap_metadata["stage2_error_vector"],
                    }

            return DecodeResult(
                success=backend_result.success,
                error_vector=backend_result.error_vector,
                objective_value=backend_result.objective_value,
                runtime_ms=backend_result.runtime_ms,
                status=backend_result.status,
                predicted_observables=predicted_observables,
                metadata=metadata,
            )
        finally:
            if owns_model_result:
                _release_solver_model(model_result.model)

    def decode_batch(
        self,
        syndromes: Sequence[Syndrome],
        *,
        weight_vector: Optional[Weights] = None,
        extra_constraints: Optional[Mapping[str, Any]] = None,
        config: Optional[DecoderConfig] = None,
    ) -> np.ndarray:
        """Decode a batch of syndromes and return predicted observables.

        Args:
            syndromes: Sequence of syndrome vectors.
            weight_vector: Optional per-variable weights.
            extra_constraints: Optional named constraint groups.
            config: Optional override for decoder configuration.

        Returns:
            2D numpy array of predicted observable flips.

        """
        results = self.decode_batch_result(
            syndromes,
            weight_vector=weight_vector,
            extra_constraints=extra_constraints,
            config=config,
        )
        if not results:
            return np.zeros((0, self._observables.shape[0]), dtype=np.int64)
        return np.stack([result.predicted_observables for result in results], axis=0)

    def decode_batch_result(
        self,
        syndromes: Sequence[Syndrome],
        *,
        weight_vector: Optional[Weights] = None,
        extra_constraints: Optional[Mapping[str, Any]] = None,
        config: Optional[DecoderConfig] = None,
        get_logicalgap: bool = False,
        get_logical_gap: Optional[bool] = None,
        get_gap_detail: bool = False,
        logical_gap_flip_last_detector: bool = False,
    ) -> Sequence[DecodeResult]:
        """Decode a batch of syndromes and return DecodeResult objects."""

        self._ensure_open()
        get_logicalgap = _resolve_logical_gap_flag(get_logicalgap, get_logical_gap)
        if get_gap_detail and not get_logicalgap:
            raise ValueError("get_gap_detail requires get_logicalgap=True.")

        active_config = config or self._config

        # Reuse the cached model unless the caller supplies overrides.
        if weight_vector is None and extra_constraints is None and config is None:
            model_result = self._base_model_result
            owns_model_result = False
        else:
            model_result = self._model_builder(
                self._deps.env,
                self._parity_check_matrix,
                self._observables,
                active_config,
                prior=self._prior,
                weight_vector=weight_vector,
                extra_constraints=extra_constraints,
            )
            owns_model_result = True

        try:
            results = []
            for syndrome in syndromes:
                # Solve each shot independently; the backend updates RHS each time.
                backend_result = self._solver_backend.solve(
                    model_result,
                    syndrome,
                    config=active_config,
                )
                predicted_observables = _compute_observable_predictions(
                    self._observables, backend_result.error_vector
                )
                metadata = dict(backend_result.metadata)
                if get_logicalgap:
                    gap_metadata = self._compute_logical_gap(
                        syndrome,
                        predicted_observables,
                        backend_result.objective_value,
                        weight_vector=weight_vector,
                        extra_constraints=extra_constraints,
                        config=active_config,
                        flip_last_detector=logical_gap_flip_last_detector,
                    )
                    metadata["logical_gap"] = gap_metadata["logical_gap"]
                    metadata["obs_flip_idx"] = gap_metadata["obs_flip_idx"]
                    if get_gap_detail:
                        metadata["gap_detail"] = {
                            "stage1_weight": float(backend_result.objective_value),
                            "stage2_weight": gap_metadata["stage2_weight"],
                            "stage2_error_vector": gap_metadata["stage2_error_vector"],
                        }
                results.append(
                    DecodeResult(
                        success=backend_result.success,
                        error_vector=backend_result.error_vector,
                        objective_value=backend_result.objective_value,
                        runtime_ms=backend_result.runtime_ms,
                        status=backend_result.status,
                        predicted_observables=predicted_observables,
                        metadata=metadata,
                    )
                )
            return results
        finally:
            if owns_model_result:
                _release_solver_model(model_result.model)

    def _compute_logical_gap(
        self,
        syndrome: Syndrome,
        logical_class: np.ndarray,
        stage1_objective: Optional[float],
        *,
        weight_vector: Optional[Weights],
        extra_constraints: Optional[Mapping[str, Any]],
        config: DecoderConfig,
        flip_last_detector: bool,
    ) -> Mapping[str, Any]:
        """Compute logical gap via a second-stage ILP solve."""

        if stage1_objective is None:
            raise AssertionError("Logical gap requested but stage-1 objective is None.")
        logical_class = np.asarray(logical_class, dtype=int).ravel() % 2
        if logical_class.size == 0:
            raise AssertionError("Logical gap requested but no logical observables are available.")

        gap_model_result = self._gap_model_builder(
            self._deps.env,
            self._parity_check_matrix,
            self._observables,
            config,
            logical_class=logical_class,
            prior=self._prior,
            weight_vector=weight_vector,
            extra_constraints=extra_constraints,
        )
        try:
            gap_syndrome = np.asarray(syndrome, dtype=int)
            if flip_last_detector:
                if gap_syndrome.size == 0:
                    raise AssertionError("Logical gap flip requested but syndrome is empty.")
                gap_syndrome = gap_syndrome.copy()
                gap_syndrome[-1] = int(gap_syndrome[-1] ^ 1)
            gap_result = self._solver_backend.solve(
                gap_model_result,
                gap_syndrome,
                config=config,
            )
            if not gap_result.success or gap_result.objective_value is None:
                raise AssertionError("Logical gap solve failed or objective unavailable.")
            z_vars_list = gap_model_result.auxiliary_vars.get("logical_xor", [])
            obs_flip_indices = [
                i for i, v in enumerate(z_vars_list) if int(round(v.X)) == 1
            ]
            return {
                "logical_gap": float(gap_result.objective_value - stage1_objective),
                "obs_flip_idx": obs_flip_indices,
                "stage2_weight": float(gap_result.objective_value),
                "stage2_error_vector": gap_result.error_vector.copy(),
            }
        finally:
            _release_solver_model(gap_model_result.model)
