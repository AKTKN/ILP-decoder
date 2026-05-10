from __future__ import annotations

import inspect
from typing import Tuple

import numpy as np
import pymatching
import stim

from beliefmatching import detector_error_model_to_check_matrices


def append_observable_detector(circuit: stim.Circuit) -> stim.Circuit:
    """Append a detector that matches OBSERVABLE_INCLUDE(0) targets.

    This assumes the OBSERVABLE_INCLUDE(0) appears at the end of the circuit so
    the rec[-k] targets remain valid when appending a new detector.
    """

    last_targets = None
    for instruction in circuit:
        if instruction.name != "OBSERVABLE_INCLUDE":
            continue
        args = instruction.gate_args_copy()
        if len(args) == 1 and int(args[0]) == 0:
            last_targets = instruction.targets_copy()

    if last_targets is None:
        raise ValueError("OBSERVABLE_INCLUDE(0) was not found in the circuit.")

    updated = circuit.copy()
    updated.append("DETECTOR", last_targets)
    return updated


def build_gap_circuit(circuit: stim.Circuit) -> stim.Circuit:
    """Return a circuit with the logical-gap detector appended."""

    return append_observable_detector(circuit)


def build_gap_dem(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Build a detector error model for the gap circuit."""

    gap_circuit = build_gap_circuit(circuit)
    return gap_circuit.detector_error_model(decompose_errors=True)


def sample_gap_shots(circuit: stim.Circuit, num_shots: int) -> np.ndarray:
    """Sample detector data from the gap circuit (detectors + observables)."""

    gap_circuit = build_gap_circuit(circuit)
    sampler = gap_circuit.compile_detector_sampler()
    return sampler.sample(num_shots, separate_observables=False, append_observables=True)


def write_gap_shot_data_file(circuit: stim.Circuit, num_shots: int, path: str) -> None:
    """Write gap-simulation shot data to a b8 file."""

    gap_circuit = build_gap_circuit(circuit)
    dem = gap_circuit.detector_error_model(decompose_errors=True)
    shot_data = gap_circuit.compile_detector_sampler().sample(
        num_shots, separate_observables=False, append_observables=True
    )
    stim.write_shot_data_file(
        data=shot_data,
        path=path,
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
        format="b8",
    )


def compute_mwpm_logical_gap(
    circuit: stim.Circuit,
    syndrome: np.ndarray,
) -> np.ndarray:
    """Compute logical gaps from MWPM using a detector-flip procedure.

    Args:
        circuit: Stim circuit. A DETECTOR for OBSERVABLE_INCLUDE(0) is appended.
        syndrome: Detector outcomes (shots x detectors) from the appended circuit.

    Returns:
        1D numpy array of logical gaps (per shot).
    """

    dem = build_gap_dem(circuit)
    mats = detector_error_model_to_check_matrices(dem)

    edge_priors = _edge_priors_from_hyperedges(mats)
    matching = _build_mwpm_from_matrices(mats, edge_priors)

    shots = np.asarray(syndrome)
    if shots.ndim == 1:
        shots = shots[None, :]
    if shots.shape[1] != dem.num_detectors:
        raise ValueError("Syndrome length must match number of detectors in DEM.")

    shots = np.asarray(shots, dtype=bool)
    _, weights = _decode_batch_with_weights(matching, shots)

    flipped = flip_last_detector(shots)
    _, weights_flipped = _decode_batch_with_weights(matching, flipped)

    return np.asarray(weights_flipped, dtype=float) - np.asarray(weights, dtype=float)


def flip_last_detector(shots: np.ndarray) -> np.ndarray:
    """Flip the last detector column in a detector sample matrix."""

    flipped = np.asarray(shots, dtype=bool).copy()
    flipped[:, -1] = ~flipped[:, -1]
    return flipped


def _decode_batch_with_weights(
    matching: pymatching.Matching,
    shots: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    signature = inspect.signature(matching.decode_batch)
    if "return_weight" in signature.parameters:
        result = matching.decode_batch(shots, return_weight=True)
    else:
        result = matching.decode_batch(shots, return_weights=True)

    if isinstance(result, tuple):
        if len(result) == 2:
            return result
        if len(result) >= 3:
            return result[0], result[1]
    raise RuntimeError("Unexpected pymatching decode_batch return signature")


def _edge_priors_from_hyperedges(mats) -> np.ndarray:
    edge_probs = np.asarray(mats.hyperedge_to_edge_matrix @ mats.priors, dtype=float)
    eps = 1e-14
    return np.clip(edge_probs, eps, 1.0 - eps)


def _build_mwpm_from_matrices(mats, edge_priors: np.ndarray) -> pymatching.Matching:
    weights = np.log((1.0 - edge_priors) / edge_priors)
    return pymatching.Matching.from_check_matrix(
        mats.edge_check_matrix,
        weights=weights,
        use_virtual_boundary_node=True,
    )
