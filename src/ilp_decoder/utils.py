"""Utility helpers for ilp_decoder."""

from __future__ import annotations

from typing import Union

import numpy as np
import stim

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



# ======= From relay_bp lib =======
def detect_data_qubits(circuit: stim.Circuit) -> list[int]:
    """Detect data qubits as those that are only measured once in a circuit.

    Warning: This is hacky and likely will only work with your typical memory circuits.
    """
    qubit_times_measured = [0 for qubit in range(circuit.num_qubits)]

    for inst in circuit:
        if inst.name.startswith("M") and not inst.gate_args_copy():
            for qubit in inst.targets_copy():
                qubit_times_measured[qubit.qubit_value] += 1

    return [
        qubit
        for qubit, times_measured in enumerate(qubit_times_measured)
        if times_measured == 1
    ]


def filter_detectors_by_basis(
    circuit: stim.Circuit,
    basis: str,
    qubits: list[int] | None = None,
) -> stim.Circuit | tuple[stim.Circuit, list[str]]:
    """Return a new circuit filtering any detectors which do not detect the specified basis for the input qubits.

    Args:
        circuit: The original circuit
        basis: "X" or "Z"
        qubits: Data qubits to inject test errors on. Should typically be data qubits. Defaults
            to automatically detected data qubits which may not be robust.

    returns:
        The filtered circuit
    """
    assert basis in ("X", "Z")

    pauli_error = "Z" if basis == "X" else "X"

    circuit = circuit.flattened()

    noiseless_circuit = circuit.without_noise()
    sampler = noiseless_circuit.compile_detector_sampler()
    reference_detectors, reference_observables = sampler.sample(
        1, separate_observables=True
    )
    reference_detectors = reference_detectors[0, :]
    reference_observables = reference_observables[0, :]
    num_detectors = len(reference_detectors)

    detector_is_sensitive = np.full(num_detectors, False, dtype=bool)

    if qubits is None:
        to_test = detect_data_qubits(noiseless_circuit)
    else:
        to_test = qubits

    to_test_set = set(to_test)

    inst_idx = 0
    while to_test:
        for qubit in to_test:
            injected_circuit = stim.Circuit()
            injected_circuit += noiseless_circuit
            injected_circuit.insert(
                inst_idx,
                stim.CircuitInstruction(f"{pauli_error}_ERROR", [qubit], [1.0]),
            )

            injected_sampler = injected_circuit.compile_detector_sampler()
            injected_detectors, injected_observables = injected_sampler.sample(
                1, separate_observables=True
            )
            injected_detectors = injected_detectors[0, :]
            injected_observables = injected_observables[0, :]

            detectors_flipped = np.where(reference_detectors != injected_detectors)
            detector_is_sensitive[detectors_flipped] = True

        to_test = []
        for inst in noiseless_circuit[inst_idx:]:
            # Is a reset we must inject errors after
            inst_idx += 1
            if inst.name.startswith("R") or inst.name.startswith("M"):
                to_test = list(to_test_set)
                break

    filtered_circuit = stim.Circuit()
    detector_idx = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            to_insert = detector_is_sensitive[detector_idx]
            detector_idx += 1
            if not to_insert:
                continue
        filtered_circuit.append(inst)
    return filtered_circuit

