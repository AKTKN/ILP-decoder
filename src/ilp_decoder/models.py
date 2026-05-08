"""Lightweight data transfer objects for ilp_decoder.

This module intentionally isolates data representations from solver dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray

Syndrome: TypeAlias = NDArray[np.int64]
Prior: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for decoder and solver behavior.

    Attributes:
        time_limit_s: Optional wall-clock timeout in seconds.
        mip_gap: Optional optimality gap for MIP solvers.
        threads: Optional solver thread count.
        log_to_console: Whether to emit solver logs to stdout.
        enable_lazy_constraints: Flag for enabling lazy constraints if supported.
        random_seed: Optional deterministic seed for solver randomness.
        solver_options: Additional solver-specific options.
    """

    time_limit_s: Optional[float] = None
    mip_gap: Optional[float] = None
    threads: Optional[int] = None
    log_to_console: bool = False
    enable_lazy_constraints: bool = False
    random_seed: Optional[int] = None
    solver_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodeResult:
    """Result of a single decoding attempt.

    Attributes:
        success: Whether a valid correction was found.
        error_vector: Predicted residual error vector (0/1 entries).
        objective_value: Objective value (e.g., error weight) if available.
        runtime_ms: End-to-end runtime in milliseconds if recorded.
        status: Solver status string if available.
        predicted_observables: Optional predicted logical observables.
        metadata: Optional extra result metadata.
    """

    success: bool
    error_vector: NDArray[np.int64]
    objective_value: Optional[float] = None
    runtime_ms: Optional[float] = None
    status: Optional[str] = None
    predicted_observables: Optional[NDArray[np.int64]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
