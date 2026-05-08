"""Utility helpers for ilp_decoder."""

from __future__ import annotations

from typing import Union

import numpy as np

try:
	from scipy import sparse
except ImportError:  # pragma: no cover - optional dependency
	sparse = None


def is_logical_error(
	observed_observables: np.ndarray,
	observables_matrix: Union[np.ndarray, "np.typing.NDArray"],
	predicted_error: np.ndarray,
) -> np.ndarray:
	"""Return whether the predicted error implies a logical error.

	Args:
		observed_observables: Observed logical flips (shape: (n_obs,) or (n_shots, n_obs)).
		observables_matrix: Observable matrix mapping errors to logical flips.
		predicted_error: Predicted error vector (shape: (n_err,) or (n_shots, n_err)).

	Returns:
		A boolean or boolean array indicating logical error(s).

	Raises:
		ValueError: If shapes do not align for multiplication/comparison.
	"""

	obs = np.asarray(observed_observables, dtype=int)
	pred = np.asarray(predicted_error, dtype=int)
	is_sparse = sparse is not None and sparse.issparse(observables_matrix)
	obs_mat = observables_matrix if is_sparse else np.asarray(observables_matrix)

	if obs_mat.ndim != 2:
		raise ValueError("observables_matrix must be 2D")
	if obs.ndim not in (1, 2):
		raise ValueError("observed_observables must be 1D or 2D")
	if pred.ndim not in (1, 2):
		raise ValueError("predicted_error must be 1D or 2D")
	if obs.ndim != pred.ndim:
		raise ValueError("observed_observables and predicted_error must have the same rank")
	if pred.shape[-1] != obs_mat.shape[1]:
		raise ValueError("predicted_error length must match observables_matrix columns")
	if obs.shape[-1] != obs_mat.shape[0]:
		raise ValueError("observed_observables length must match observables_matrix rows")
	if obs.ndim == 2 and obs.shape[0] != pred.shape[0]:
		raise ValueError("batch size mismatch between observed_observables and predicted_error")

	if pred.ndim == 1:
		predicted_obs = obs_mat @ pred
		predicted_obs = np.asarray(predicted_obs, dtype=int) % 2
		observed_obs = obs % 2
		return np.any(predicted_obs != observed_obs)

	predicted_obs = obs_mat @ pred.T
	predicted_obs = np.asarray(predicted_obs, dtype=int).T % 2
	observed_obs = obs % 2
	return np.any(predicted_obs != observed_obs, axis=1)
