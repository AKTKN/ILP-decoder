"""Post-processing and evaluation utilities for decoder results."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .models import DecodeResult


def evaluate_logical_error_rate(
	results: Sequence[DecodeResult],
	true_errors: NDArray[np.int64],
	observables: NDArray[np.int64],
) -> float:
	"""Compute the logical error rate (LER) from a batch of results.

	Args:
		results: Decoder outcomes for each shot.
		true_errors: Ground-truth physical error vectors.
		observables: Logical observables used to evaluate logical failures.

	Returns:
		The logical error rate as a float in [0, 1].
	"""

	pass


def verify_stabilizer_condition(
	residual_error: NDArray[np.int64],
	parity_check_matrix: NDArray[np.int64],
) -> bool:
	"""Check whether a residual error satisfies stabilizer constraints.

	Args:
		residual_error: Residual error vector after correction.
		parity_check_matrix: Parity check matrix defining the code space.

	Returns:
		True if the stabilizer condition is satisfied, otherwise False.
	"""

	pass
