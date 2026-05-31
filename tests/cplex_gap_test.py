"""CPLEX gap tests.

These tests mirror gurobi_gap_test.py but use the CPLEX backend.

Licence detection
-----------------
The bivariate bicycle code tests are large problems.  ``_require_full_cplex_license()``
probes CPLEX at runtime; tests skip automatically on a community-edition install
and run in full on an academic licence.
"""

import os
import sys

import numpy as np
import pytest
import stim
from scipy.sparse import csc_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from beliefmatching import detector_error_model_to_check_matrices
from ilp_decoder import DecoderConfig, ILPDecoder
from ilp_decoder.utils import filter_detectors_by_basis, is_logical_error
from ilp_decoder.core import DecoderDependencies, OptimizerBackend
from ilp_decoder.cplex_backend import CplexEnv
from ilp_decoder.cplex_formulation import build_logical_gap_model_cplex

test_dir = os.path.dirname(os.path.realpath(__file__))

H_STEANE = csc_matrix(
    [
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1],
    ]
)
LX_STEANE = csc_matrix([[1, 1, 1, 1, 1, 1, 1]])

circuit_path = {
    "bb_18_4_3": "circuit=bicycle_bivariate_18_4_3_memory_Z,distance=3,rounds=3,error_rate=0.005,noise_model=uniform_circuit,basis=CX,A=x+1+y,B=x+1+xy^2.stim",
    "bb_72_12_6": "circuit=bicycle_bivariate_72_12_6_memory_Z,distance=6,rounds=6,error_rate=0.005,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim",
    "bb_144_12_12": "circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate=0.005,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim",
}

TEST_CODE_PARAM = "bb_72_12_6"
BIVARIATE_BASIS = "Z"
BIVARIATE_NUM_SHOTS = 20


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
    """Skip the calling test if CPLEX community-edition limit is in effect."""
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
# Steane-code gap tests (always safe: 10 variables)
# ---------------------------------------------------------------------------

def test_cplex_steane_logical_gap_non_negative(cplex_env):
    """Logical gap must be >= 0 for all Steane code shots."""
    prior = np.full(7, 0.1, dtype=float)
    decoder = _build_ilp_decoder(cplex_env, H_STEANE, LX_STEANE, prior)

    test_errors = [
        np.array([1, 0, 0, 0, 0, 0, 0], dtype=int),
        np.array([0, 1, 0, 0, 0, 0, 0], dtype=int),
        np.array([0, 0, 0, 0, 0, 0, 0], dtype=int),
    ]
    syndromes = [np.asarray((H_STEANE @ e) % 2, dtype=int).ravel() for e in test_errors]

    results = decoder.decode_batch_result(syndromes, get_logicalgap=True)
    for i, result in enumerate(results):
        assert result.success, f"Shot {i}: solve failed"
        gap = result.metadata.get("logical_gap")
        assert gap is not None, f"Shot {i}: logical_gap missing"
        assert gap >= 0.0, f"Shot {i}: negative gap {gap}"


def test_cplex_steane_obs_flip_idx(cplex_env):
    """obs_flip_idx must be non-empty and match an independent stage-2 solve."""
    prior = np.full(7, 0.1, dtype=float)
    decoder = _build_ilp_decoder(cplex_env, H_STEANE, LX_STEANE, prior)

    error_vector = np.array([1, 0, 0, 0, 0, 0, 0], dtype=int)
    syndrome = np.asarray((H_STEANE @ error_vector) % 2, dtype=int).ravel()

    result = decoder.decode_result(syndrome, get_logicalgap=True)
    assert result.success

    flip_idx = result.metadata.get("obs_flip_idx")
    assert isinstance(flip_idx, list)
    assert len(flip_idx) > 0
    assert all(0 <= idx < LX_STEANE.shape[0] for idx in flip_idx)

    # Independent stage-2 solve.
    config = DecoderConfig(log_to_console=False)
    logical_class = np.asarray(result.predicted_observables, dtype=int).ravel() % 2
    gap_result = build_logical_gap_model_cplex(
        cplex_env, H_STEANE, LX_STEANE, config,
        logical_class=logical_class, prior=prior,
    )
    for j, constr in enumerate(gap_result.syndrome_constraints):
        constr.RHS = int(syndrome[j])
    gap_result.model.set_log_stream(None)
    gap_result.model.set_results_stream(None)
    gap_result.model.set_warning_stream(None)
    gap_result.model.set_error_stream(None)
    gap_result.model.solve()

    status = gap_result.model.solution.get_status()
    assert status in {101, 102, 1, 5}, f"Reference solve status {status}"

    z_vars_list = gap_result.auxiliary_vars["logical_xor"]
    expected = sorted(k for k, v in enumerate(z_vars_list) if int(round(v.X)) == 1)
    assert sorted(flip_idx) == expected


# ---------------------------------------------------------------------------
# Bivariate bicycle code tests (require academic CPLEX licence)
# ---------------------------------------------------------------------------

def _selected_circuit_filename():
    if TEST_CODE_PARAM not in circuit_path:
        raise KeyError(f"Unknown TEST_CODE_PARAM: {TEST_CODE_PARAM}")
    return circuit_path[TEST_CODE_PARAM]


def _bbcode_artifact_paths(circuit_filename, basis, num_shots):
    stem = os.path.splitext(os.path.basename(circuit_filename))[0]
    basis_tag = f"{basis}_filtered"
    dem_path = os.path.join(test_dir, f"{stem}_{basis_tag}.dem")
    shots_path = os.path.join(test_dir, f"{stem}_{basis_tag}_{num_shots}_shots.b8")
    return dem_path, shots_path


@pytest.fixture(scope="module")
def bbcode_data():
    circuit_filename = _selected_circuit_filename()
    circuit = stim.Circuit.from_file(os.path.join(test_dir, circuit_filename))
    filtered_circuit = filter_detectors_by_basis(circuit, BIVARIATE_BASIS)
    dem_path, shots_path = _bbcode_artifact_paths(
        circuit_filename, BIVARIATE_BASIS, BIVARIATE_NUM_SHOTS
    )

    if os.path.exists(dem_path):
        dem = stim.DetectorErrorModel.from_file(dem_path)
    else:
        dem = filtered_circuit.detector_error_model(decompose_errors=False)
        dem.to_file(dem_path)

    if os.path.exists(shots_path):
        shot_data = stim.read_shot_data_file(
            path=shots_path, format="b8",
            num_detectors=dem.num_detectors, num_observables=dem.num_observables,
        )
    else:
        sampler = filtered_circuit.compile_detector_sampler()
        shot_data = sampler.sample(
            BIVARIATE_NUM_SHOTS, separate_observables=False, append_observables=True
        )
        stim.write_shot_data_file(
            data=shot_data, path=shots_path,
            num_detectors=dem.num_detectors, num_observables=dem.num_observables,
            format="b8",
        )

    mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    shots = shot_data[:, 0: dem.num_detectors]
    observables = shot_data[:, dem.num_detectors:]
    return {"mats": mats, "priors": mats.priors, "shots": shots, "observables": observables}


def test_cplex_bbcode_ilp_decoder_end_to_end(cplex_env, bbcode_data):
    _require_full_cplex_license()

    mats = bbcode_data["mats"]
    priors = bbcode_data["priors"]
    shots = bbcode_data["shots"]
    observables = bbcode_data["observables"]

    decoder = _build_ilp_decoder(
        cplex_env, mats.check_matrix, mats.observables_matrix, priors
    )
    ilp_results = decoder.decode_batch_result(shots)
    assert all(r.success for r in ilp_results)

    predicted_observables = np.stack([r.predicted_observables for r in ilp_results], axis=0)
    assert predicted_observables.shape == observables.shape

    predicted_errors = np.stack([r.error_vector for r in ilp_results], axis=0)
    logical_errors = is_logical_error(observables, mats.observables_matrix, predicted_errors)
    logical_errors = np.asarray(logical_errors)
    assert logical_errors.shape[0] == shots.shape[0]


def test_cplex_bbcode_ilp_decoder_logical_gap_end_to_end(cplex_env, bbcode_data):
    _require_full_cplex_license()

    mats = bbcode_data["mats"]
    priors = bbcode_data["priors"]
    shots = bbcode_data["shots"]

    decoder = _build_ilp_decoder(
        cplex_env, mats.check_matrix, mats.observables_matrix, priors
    )
    ilp_results = decoder.decode_batch_result(
        shots, get_logicalgap=True, logical_gap_flip_last_detector=False,
    )
    ilp_gap_values = [r.metadata.get("logical_gap") for r in ilp_results]
    assert all(gap is not None for gap in ilp_gap_values)
    ilp_gaps = np.array([float(gap) for gap in ilp_gap_values], dtype=float)
    assert ilp_gaps.shape[0] == shots.shape[0]


def test_cplex_bbcode_obs_flip_idx(cplex_env, bbcode_data):
    _require_full_cplex_license()

    mats = bbcode_data["mats"]
    priors = bbcode_data["priors"]
    shots = bbcode_data["shots"]
    num_observables = mats.observables_matrix.shape[0]
    assert num_observables >= 4

    decoder = _build_ilp_decoder(
        cplex_env, mats.check_matrix, mats.observables_matrix, priors
    )
    ilp_results = decoder.decode_batch_result(
        shots, get_logicalgap=True, logical_gap_flip_last_detector=False,
    )

    config = DecoderConfig(log_to_console=False)

    for i, result in enumerate(ilp_results):
        assert "obs_flip_idx" in result.metadata, f"Shot {i}: obs_flip_idx missing"
        flip_idx = result.metadata["obs_flip_idx"]

        assert isinstance(flip_idx, list), f"Shot {i}: obs_flip_idx is not a list"
        assert len(flip_idx) > 0, f"Shot {i}: obs_flip_idx is empty (violates sum(z)>=1)"
        assert all(0 <= idx < num_observables for idx in flip_idx), (
            f"Shot {i}: obs_flip_idx contains out-of-range index"
        )

        logical_class = np.asarray(result.predicted_observables, dtype=int).ravel() % 2
        gap_model_result = build_logical_gap_model_cplex(
            cplex_env, mats.check_matrix, mats.observables_matrix, config,
            logical_class=logical_class, prior=priors,
        )
        gap_model_result.model.set_log_stream(None)
        gap_model_result.model.set_results_stream(None)
        gap_model_result.model.set_warning_stream(None)
        gap_model_result.model.set_error_stream(None)
        for j, constr in enumerate(gap_model_result.syndrome_constraints):
            constr.RHS = int(shots[i][j])
        gap_model_result.model.solve()

        status = gap_model_result.model.solution.get_status()
        assert status in {101, 102, 1, 5}, f"Shot {i}: reference solve status {status}"

        z_vars_list = gap_model_result.auxiliary_vars["logical_xor"]
        expected = sorted(
            k for k, v in enumerate(z_vars_list) if int(round(v.X)) == 1
        )
        assert sorted(flip_idx) == expected, (
            f"Shot {i}: obs_flip_idx {sorted(flip_idx)} != reference z_vars {expected}"
        )
