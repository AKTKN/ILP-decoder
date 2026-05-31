"""CPLEX decoder tests.

These tests mirror gurobi_decoder_test.py but use the CPLEX backend.

Licence detection
-----------------
Tests whose problem size may exceed the community-edition limit (1000 vars)
are guarded by ``_require_full_cplex_license()``, which probes CPLEX directly.
With an academic licence all tests run; without one the large-problem tests
skip automatically with an explanatory message.
"""

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
from ilp_decoder.core import DecoderDependencies, OptimizerBackend
from ilp_decoder.cplex_backend import CplexEnv
from ilp_decoder.gap_mwpm import (
    build_gap_dem,
    compute_mwpm_logical_gap,
    write_gap_shot_data_file,
)

test_dir = os.path.dirname(os.path.realpath(__file__))

H_STEANE = csc_matrix(
    [
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1],
    ]
)
LX_STEANE = csc_matrix([[1, 1, 1, 1, 1, 1, 1]])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cplex_env():
    pytest.importorskip("cplex")
    return CplexEnv()


def _build_ilp_decoder(env, parity_check_matrix, observables, prior) -> ILPDecoder:
    config = DecoderConfig(log_to_console=False)
    deps = DecoderDependencies(env=env, backend=OptimizerBackend.CPLEX)
    return ILPDecoder(
        parity_check_matrix=parity_check_matrix,
        observables=observables,
        prior=prior,
        config=config,
        deps=deps,
    )


# ---------------------------------------------------------------------------
# Licence detection
# ---------------------------------------------------------------------------

def _require_full_cplex_license():
    """Skip the calling test if the CPLEX community-edition limit is in effect."""
    import cplex
    try:
        c = cplex.Cplex()
        c.set_log_stream(None)
        c.set_results_stream(None)
        c.set_warning_stream(None)
        c.set_error_stream(None)
        c.variables.add(
            names=[f"x{i}" for i in range(1001)],
            types=["B"] * 1001,
        )
        c.end()
    except cplex.exceptions.CplexError as exc:
        pytest.skip(f"Full CPLEX licence required to run this test: {exc}")


# ---------------------------------------------------------------------------
# Helper: shared shot data generation for the surface-code tests
# ---------------------------------------------------------------------------

def _load_surface_code_matrices():
    circuit = stim.Circuit.from_file(
        os.path.join(test_dir, "surface_code_rotated_memory_x_d_7_p_0.007.stim")
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    mats = detector_error_model_to_check_matrices(dem)
    eps = 1e-14
    edge_probs = np.asarray(mats.hyperedge_to_edge_matrix @ mats.priors, dtype=float)
    edge_priors = np.clip(edge_probs, eps, 1.0 - eps)
    return circuit, dem, mats, edge_priors


def _decode_batch_with_weights(matching: pymatching.Matching, shots: np.ndarray):
    sig = inspect.signature(matching.decode_batch)
    if "return_weight" in sig.parameters:
        result = matching.decode_batch(shots, return_weight=True)
    elif "return_weights" in sig.parameters:
        result = matching.decode_batch(shots, return_weights=True)
    else:
        raise RuntimeError("pymatching decode_batch does not expose weight return flags")
    if isinstance(result, tuple):
        return result[0], result[1]
    raise RuntimeError("Unexpected pymatching decode_batch return signature")


def _build_mwpm(mats, edge_priors):
    weights = np.log((1.0 - edge_priors) / edge_priors)
    return pymatching.Matching.from_check_matrix(
        mats.edge_check_matrix,
        weights=weights,
        use_virtual_boundary_node=True,
    )


# ---------------------------------------------------------------------------
# Tests: always run (Steane code, 10 variables)
# ---------------------------------------------------------------------------

def test_cplex_env_type_validation():
    """Passing a mismatched env/backend pair must raise TypeError."""
    pytest.importorskip("cplex")
    pytest.importorskip("gurobipy")

    import gurobipy as gp

    try:
        gp_env = gp.Env(empty=True)
        gp_env.setParam("OutputFlag", 0)
        gp_env.start()
    except gp.GurobiError:
        pytest.skip("Gurobi not available")

    prior = np.full(7, 0.1, dtype=float)
    config = DecoderConfig(log_to_console=False)

    with pytest.raises(TypeError, match="CPLEX backend requires a CplexEnv"):
        deps = DecoderDependencies(env=gp_env, backend=OptimizerBackend.CPLEX)
        ILPDecoder(
            parity_check_matrix=H_STEANE, observables=LX_STEANE,
            prior=prior, config=config, deps=deps,
        )

    gp_env.dispose()

    with pytest.raises(TypeError, match="GUROBI backend does not accept a CplexEnv"):
        deps = DecoderDependencies(env=CplexEnv(), backend=OptimizerBackend.GUROBI)
        ILPDecoder(
            parity_check_matrix=H_STEANE, observables=LX_STEANE,
            prior=prior, config=config, deps=deps,
        )


def test_cplex_ilp_decoder_steane_end_to_end(cplex_env):
    prior = np.full(7, 0.1, dtype=float)
    error_vector = np.array([1, 0, 0, 0, 0, 0, 0], dtype=int)
    syndrome = np.asarray((H_STEANE @ error_vector) % 2, dtype=int).ravel()

    decoder = _build_ilp_decoder(cplex_env, H_STEANE, LX_STEANE, prior)
    result = decoder.decode_result(syndrome)

    assert result.success
    assert np.array_equal(result.error_vector, error_vector)

    weights = np.log((1.0 - prior) / prior)
    expected_objective = float(weights @ error_vector)
    assert result.objective_value is not None
    assert np.isclose(result.objective_value, expected_objective, rtol=1e-6, atol=1e-8)

    expected_observables = np.asarray((LX_STEANE @ error_vector) % 2, dtype=int).ravel()
    assert np.array_equal(result.predicted_observables, expected_observables)


def test_cplex_ilp_decoder_steane_batch(cplex_env):
    prior = np.full(7, 0.1, dtype=float)
    weights = np.log((1.0 - prior) / prior)

    decoder = _build_ilp_decoder(cplex_env, H_STEANE, LX_STEANE, prior)

    test_cases = [
        np.array([1, 0, 0, 0, 0, 0, 0], dtype=int),
        np.array([0, 1, 0, 0, 0, 0, 0], dtype=int),
        np.array([0, 0, 0, 1, 0, 0, 0], dtype=int),
    ]
    syndromes = [np.asarray((H_STEANE @ e) % 2, dtype=int).ravel() for e in test_cases]

    results = decoder.decode_batch_result(syndromes)
    assert len(results) == len(test_cases)
    for i, result in enumerate(results):
        assert result.success, f"Shot {i}: solve failed"
        expected_obj = float(weights @ test_cases[i])
        assert np.isclose(result.objective_value, expected_obj, rtol=1e-6, atol=1e-8), (
            f"Shot {i}: objective mismatch"
        )


def test_cplex_ilp_decoder_steane_logical_gap(cplex_env):
    prior = np.full(7, 0.1, dtype=float)
    error_vector = np.array([1, 0, 0, 0, 0, 0, 0], dtype=int)
    syndrome = np.asarray((H_STEANE @ error_vector) % 2, dtype=int).ravel()

    decoder = _build_ilp_decoder(cplex_env, H_STEANE, LX_STEANE, prior)
    result = decoder.decode_result(syndrome, get_logicalgap=True)

    assert result.success
    assert "logical_gap" in result.metadata
    assert "obs_flip_idx" in result.metadata

    gap = result.metadata["logical_gap"]
    assert gap is not None
    assert gap >= 0.0

    flip_idx = result.metadata["obs_flip_idx"]
    assert isinstance(flip_idx, list)
    assert len(flip_idx) > 0


def test_cplex_solver_metadata(cplex_env):
    prior = np.full(7, 0.1, dtype=float)
    syndrome = np.zeros(3, dtype=int)

    decoder = _build_ilp_decoder(cplex_env, H_STEANE, LX_STEANE, prior)
    result = decoder.decode_result(syndrome)

    assert result.metadata.get("solver") == "cplex"
    assert "status_code" in result.metadata


# ---------------------------------------------------------------------------
# Tests: require full (academic) CPLEX licence
# ---------------------------------------------------------------------------

def test_cplex_ilp_decoder_surface_code_against_mwpm(cplex_env):
    """Surface code d=7: compare CPLEX ILP objectives against MWPM."""
    _require_full_cplex_license()

    shot_path = os.path.join(
        test_dir, "surface_code_rotated_memory_x_d_7_p_0.007_50_shots.b8"
    )
    if not os.path.exists(shot_path):
        # Generate shot data if not already present.
        d, p, num_shots = 7, 0.007, 50
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            rounds=d, distance=d,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
            after_clifford_depolarization=p,
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        sampler = circuit.compile_detector_sampler()
        shot_data = sampler.sample(num_shots, separate_observables=False, append_observables=True)
        stim.write_shot_data_file(
            data=shot_data, path=shot_path,
            num_detectors=dem.num_detectors, num_observables=dem.num_observables,
            format="b8",
        )

    circuit, dem, mats, edge_priors = _load_surface_code_matrices()

    decoder = _build_ilp_decoder(
        cplex_env, mats.edge_check_matrix, mats.edge_observables_matrix, edge_priors
    )

    shot_data = stim.read_shot_data_file(
        path=shot_path, format="b8",
        num_detectors=dem.num_detectors, num_observables=dem.num_observables,
    )
    shots = shot_data[:, 0: dem.num_detectors]
    observables = shot_data[:, dem.num_detectors:]

    matching = _build_mwpm(mats, edge_priors)
    mwpm_predicted, mwpm_weights = _decode_batch_with_weights(matching, shots)

    ilp_results = decoder.decode_batch_result(shots)
    assert all(r.success for r in ilp_results)

    ilp_weights = np.array([float(r.objective_value) for r in ilp_results], dtype=float)
    assert np.allclose(ilp_weights, mwpm_weights, rtol=1e-6, atol=1e-8)

    ilp_logical_errors = np.array([
        is_logical_error(observables[i], mats.edge_observables_matrix, r.error_vector)
        for i, r in enumerate(ilp_results)
    ])
    mwpm_logical_errors = is_logical_error(
        observables, mats.edge_observables_matrix, mwpm_predicted
    )
    assert np.array_equal(ilp_logical_errors, mwpm_logical_errors)


def test_cplex_gap_collection(cplex_env):
    """Surface code d=7 gap: compare CPLEX ILP gaps against MWPM gaps."""
    _require_full_cplex_license()

    gap_shot_path = os.path.join(
        test_dir, "surface_code_rotated_memory_x_d_7_p_0.007_50_shots_gapsim.b8"
    )
    circuit, _, _, _ = _load_surface_code_matrices()
    # Reload circuit since _load_surface_code_matrices returns the base dem.
    circuit = stim.Circuit.from_file(
        os.path.join(test_dir, "surface_code_rotated_memory_x_d_7_p_0.007.stim")
    )

    if not os.path.exists(gap_shot_path):
        write_gap_shot_data_file(circuit, 50, gap_shot_path)

    dem = build_gap_dem(circuit)
    mats = detector_error_model_to_check_matrices(dem)
    eps = 1e-14
    edge_probs = np.asarray(mats.hyperedge_to_edge_matrix @ mats.priors, dtype=float)
    edge_priors = np.clip(edge_probs, eps, 1.0 - eps)

    decoder = _build_ilp_decoder(
        cplex_env, mats.edge_check_matrix, mats.edge_observables_matrix, edge_priors
    )

    shot_data = stim.read_shot_data_file(
        path=gap_shot_path, format="b8",
        num_detectors=dem.num_detectors, num_observables=dem.num_observables,
    )
    shots = shot_data[:, 0: dem.num_detectors]
    shots = shots[:20]

    ilp_results = decoder.decode_batch_result(
        shots,
        get_logicalgap=True,
        logical_gap_flip_last_detector=True,
    )
    ilp_gap_values = [r.metadata.get("logical_gap") for r in ilp_results]

    assert all(gap is not None for gap in ilp_gap_values)
    ilp_gaps = np.array([float(g) for g in ilp_gap_values], dtype=float)

    mwpm_gaps = compute_mwpm_logical_gap(circuit, shots)

    assert ilp_gaps.shape == mwpm_gaps.shape
    assert np.allclose(ilp_gaps, mwpm_gaps, rtol=0.0, atol=1e-4)

    matching = _build_mwpm(mats, edge_priors)
    mwpm_predicted, _ = _decode_batch_with_weights(matching, shots)
    weights = np.asarray(np.log((1.0 - edge_priors) / edge_priors), dtype=np.float64)
    ilp_objectives = np.array(
        [float(weights @ r.error_vector) for r in ilp_results], dtype=float
    )
    mwpm_objectives = np.array(
        [float(weights @ pred) for pred in mwpm_predicted], dtype=float
    )
    assert np.allclose(ilp_objectives, mwpm_objectives, rtol=0.0, atol=1e-8)
