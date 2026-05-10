import inspect
import os
import sys

import numpy as np
import pytest
import pymatching
import stim
from scipy.sparse import csc_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from beliefmatching import detector_error_model_to_check_matrices
from ilp_decoder import DecoderConfig, ILPDecoder
from ilp_decoder.utils import is_logical_error
from ilp_decoder.gap_mwpm import (
    build_gap_dem,
    compute_mwpm_logical_gap,
    write_gap_shot_data_file,
)
from ilp_decoder.core import DecoderDependencies

test_dir = os.path.dirname(os.path.realpath(__file__))

H_STEANE = csc_matrix(
    [
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1],
    ]
)
LX_STEANE = csc_matrix([[1, 1, 1, 1, 1, 1, 1]])


@pytest.fixture(scope="module")
def gurobi_env():
    gp = pytest.importorskip("gurobipy")
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi not available: {exc}")
    yield env
    env.dispose()

def generate_shot_data():
    d = 7
    p = 0.007
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=d,
        distance=d,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        after_clifford_depolarization=p,
    )
    circuit.to_file(f"surface_code_rotated_memory_x_d_{d}_p_{p}.stim")
    dem = circuit.detector_error_model(decompose_errors=True)
    dem.to_file(f"surface_code_rotated_memory_x_d_{d}_p_{p}.dem")

    num_shots = 50

    sampler = circuit.compile_detector_sampler()
    shot_data = sampler.sample(
        num_shots, separate_observables=False, append_observables=True
    )
    stim.write_shot_data_file(
        data=shot_data,
        path=f"surface_code_rotated_memory_x_d_{d}_p_{p}_{num_shots}_shots.b8",
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
        format="b8",
    )


def _build_ilp_decoder(env, parity_check_matrix, observables, prior) -> ILPDecoder:
    config = DecoderConfig(log_to_console=False)
    deps = DecoderDependencies(env=env)
    return ILPDecoder(
        parity_check_matrix=parity_check_matrix,
        observables=observables,
        prior=prior,
        config=config,
        deps=deps,
    )


def _decode_batch_with_weights(matching: pymatching.Matching, shots: np.ndarray):
    signature = inspect.signature(matching.decode_batch)
    if "return_weight" in signature.parameters:
        result = matching.decode_batch(shots, return_weight=True)
    elif "return_weights" in signature.parameters:
        result = matching.decode_batch(shots, return_weights=True)
    else:
        raise RuntimeError("pymatching decode_batch does not expose weight return flags")

    if isinstance(result, tuple):
        if len(result) == 2:
            return result
        if len(result) >= 3:
            return result[0], result[1]
    raise RuntimeError("Unexpected pymatching decode_batch return signature")


def _edge_priors_from_hyperedges(mats):
    edge_probs = np.asarray(mats.hyperedge_to_edge_matrix @ mats.priors, dtype=float)
    eps = 1e-14
    return np.clip(edge_probs, eps, 1.0 - eps)


def _build_mwpm_from_matrices(mats, edge_priors):
    weights = np.log((1.0 - edge_priors) / edge_priors)
    return pymatching.Matching.from_check_matrix(
        mats.edge_check_matrix,
        weights=weights,
        use_virtual_boundary_node=True,
    )


def test_ilp_decoder_steane_end_to_end(gurobi_env):
    prior = np.full(7, 0.1, dtype=float)
    error_vector = np.array([1, 0, 0, 0, 0, 0, 0], dtype=int)
    syndrome = np.asarray((H_STEANE @ error_vector) % 2, dtype=int).ravel()

    decoder = _build_ilp_decoder(gurobi_env, H_STEANE, LX_STEANE, prior)
    result = decoder.decode_result(syndrome)

    assert result.success
    assert np.array_equal(result.error_vector, error_vector)

    weights = np.log((1.0 - prior) / prior)
    expected_objective = float(weights @ error_vector)
    assert result.objective_value is not None
    assert np.isclose(result.objective_value, expected_objective, rtol=1e-6, atol=1e-8)

    expected_observables = np.asarray((LX_STEANE @ error_vector) % 2, dtype=int).ravel()
    assert np.array_equal(result.predicted_observables, expected_observables)


def test_ilp_decoder_surface_code_against_mwpm(gurobi_env):

    dem_path = os.path.join(
            test_dir, "surface_code_rotated_memory_x_d_7_p_0.007_50_shots.b8"
        )
    if not os.path.exists("surface_code_rotated_memory_x_d_7_p_0.007_50_shots.b8"):
        generate_shot_data()
    
    circuit = stim.Circuit.from_file(
        os.path.join(test_dir, "surface_code_rotated_memory_x_d_7_p_0.007.stim")
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    mats = detector_error_model_to_check_matrices(dem)
    edge_priors = _edge_priors_from_hyperedges(mats)

    decoder = _build_ilp_decoder(
        gurobi_env, mats.edge_check_matrix, mats.edge_observables_matrix, edge_priors
    )

    shot_data = stim.read_shot_data_file(
        path=os.path.join(
            test_dir, "surface_code_rotated_memory_x_d_7_p_0.007_50_shots.b8"
        ),
        format="b8",
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
    )

    shots = shot_data[:, 0 : dem.num_detectors]
    observables = shot_data[:, dem.num_detectors :]

    matching = _build_mwpm_from_matrices(mats, edge_priors)
    mwpm_predicted, mwpm_weights = _decode_batch_with_weights(matching, shots)

    ilp_results = decoder.decode_batch_result(shots)
    assert all(result.success for result in ilp_results)
    ilp_weights = np.array(
        [float(result.objective_value) for result in ilp_results], dtype=float
    )
    assert np.allclose(ilp_weights, mwpm_weights, rtol=1e-6, atol=1e-8)

    ilp_logical_errors = np.array(
        [
            is_logical_error(observables[i], mats.edge_observables_matrix, result.error_vector)
            for i, result in enumerate(ilp_results)
        ]
    )
    mwpm_logical_errors = is_logical_error(
        observables, mats.edge_observables_matrix, mwpm_predicted
    )
    assert np.array_equal(ilp_logical_errors, mwpm_logical_errors)


def test_gap_collection(gurobi_env):

    dem_path = os.path.join(
        test_dir, "surface_code_rotated_memory_x_d_7_p_0.007_50_shots_gapsim.b8"
    )

    circuit = stim.Circuit.from_file(
        os.path.join(test_dir, "surface_code_rotated_memory_x_d_7_p_0.007.stim")
    )

    if not os.path.exists(dem_path):
        write_gap_shot_data_file(circuit, 50, dem_path)

    dem = build_gap_dem(circuit)
    mats = detector_error_model_to_check_matrices(dem)
    edge_priors = _edge_priors_from_hyperedges(mats)

    decoder = _build_ilp_decoder(
        gurobi_env, mats.edge_check_matrix, mats.edge_observables_matrix, edge_priors
    )

    shot_data = stim.read_shot_data_file(
        path=dem_path,
        format="b8",
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
    )
    shots = shot_data[:, 0 : dem.num_detectors]
    shots = shots[:20]

    ilp_results = decoder.decode_batch_result(
        shots,
        get_logicalgap=True,
        logical_gap_flip_last_detector=True,
    )
    ilp_gap_values = [result.metadata.get("logical_gap") for result in ilp_results]

    assert all(gap is not None for gap in ilp_gap_values)
    ilp_gaps = np.array([float(gap) for gap in ilp_gap_values], dtype=float)

    mwpm_gaps = compute_mwpm_logical_gap(circuit, shots)

    print("ILP gaps:", ilp_gaps)
    print("MWPM gaps:", mwpm_gaps)

    assert ilp_gaps.shape == mwpm_gaps.shape
    assert np.allclose(ilp_gaps, mwpm_gaps, rtol=0.0, atol=1e-4)

    matching = _build_mwpm_from_matrices(mats, edge_priors)
    mwpm_predicted, _ = _decode_batch_with_weights(matching, shots)
    weights = np.asarray(np.log((1.0 - edge_priors) / edge_priors), dtype=np.float64)
    ilp_objectives = np.array(
        [float(weights @ result.error_vector) for result in ilp_results], dtype=float
    )
    mwpm_objectives = np.array(
        [float(weights @ pred) for pred in mwpm_predicted], dtype=float
    )
    assert ilp_objectives.shape == mwpm_objectives.shape
    assert np.allclose(ilp_objectives, mwpm_objectives, rtol=0.0, atol=1e-8)


# Notes on MWPM vs ILP alignment:
# - The ILP decoder in this file supports hyperedge variables (from DEM) or edge variables.
# - MWPM operates on edges, so to make objective values comparable we construct MWPM from
#   edge check matrices and feed it the same edge-level priors as the ILP.
# - We convert edge priors to weights via log-odds: w = log((1 - p) / p). This matches
#   the ILP objective definition in the decoder and avoids mixing weight conventions
#   (e.g., -log(p) vs log-odds).
# - If you instead build MWPM directly from the DEM (or use hyperedge variables for ILP),
#   the objective values need not match even when the logical error rates are similar.

