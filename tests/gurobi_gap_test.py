import os
import sys

import numpy as np
import pytest
import stim

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from beliefmatching import detector_error_model_to_check_matrices
from ilp_decoder import DecoderConfig, ILPDecoder
from ilp_decoder.utils import filter_detectors_by_basis, is_logical_error
from ilp_decoder.core import DecoderDependencies

test_dir = os.path.dirname(os.path.realpath(__file__))

circuit_path = {
    "bb_18_4_3": "circuit=bicycle_bivariate_18_4_3_memory_Z,distance=3,rounds=3,error_rate=0.005,noise_model=uniform_circuit,basis=CX,A=x+1+y,B=x+1+xy^2.stim",
    "bb_72_12_6": "circuit=bicycle_bivariate_72_12_6_memory_Z,distance=6,rounds=6,error_rate=0.005,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim",
    "bb_144_12_12": "circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate=0.005,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim",
}


TEST_CODE_PARAM = "bb_72_12_6"  # Change this to test different circuits from the above dictionary.
BIVARIATE_BASIS = "Z"
BIVARIATE_NUM_SHOTS = 20


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


def _selected_circuit_filename() -> str:
    if TEST_CODE_PARAM not in circuit_path:
        raise KeyError(f"Unknown TEST_CODE_PARAM: {TEST_CODE_PARAM}")
    return circuit_path[TEST_CODE_PARAM]


def _bbcode_file_stem(circuit_filename: str) -> str:
    return os.path.splitext(os.path.basename(circuit_filename))[0]


def _bbcode_artifact_paths(
    circuit_filename: str, basis: str, num_shots: int
) -> tuple[str, str]:
    stem = _bbcode_file_stem(circuit_filename)
    basis_tag = f"{basis}_filtered"
    dem_path = os.path.join(test_dir, f"{stem}_{basis_tag}.dem")
    shots_path = os.path.join(
        test_dir, f"{stem}_{basis_tag}_{num_shots}_shots.b8"
    )
    return dem_path, shots_path


def _load_bbcode_circuit() -> stim.Circuit:
    circuit_filename = _selected_circuit_filename()
    return stim.Circuit.from_file(os.path.join(test_dir, circuit_filename))


def _load_or_create_dem(filtered_circuit: stim.Circuit, dem_path: str) -> stim.DetectorErrorModel:
    if os.path.exists(dem_path):
        return stim.DetectorErrorModel.from_file(dem_path)
    dem = filtered_circuit.detector_error_model(decompose_errors=False)
    dem.to_file(dem_path)
    return dem


def _load_or_create_shots(
    filtered_circuit: stim.Circuit,
    dem: stim.DetectorErrorModel,
    shots_path: str,
    num_shots: int,
) -> np.ndarray:
    if os.path.exists(shots_path):
        return stim.read_shot_data_file(
            path=shots_path,
            format="b8",
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
        )
    sampler = filtered_circuit.compile_detector_sampler()
    shot_data = sampler.sample(
        num_shots, separate_observables=False, append_observables=True
    )
    stim.write_shot_data_file(
        data=shot_data,
        path=shots_path,
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
        format="b8",
    )
    return shot_data


@pytest.fixture(scope="module")
def bbcode_data():
    circuit_filename = _selected_circuit_filename()
    circuit = _load_bbcode_circuit()
    filtered_circuit = filter_detectors_by_basis(circuit, BIVARIATE_BASIS)
    dem_path, shots_path = _bbcode_artifact_paths(
        circuit_filename, BIVARIATE_BASIS, BIVARIATE_NUM_SHOTS
    )
    dem = _load_or_create_dem(filtered_circuit, dem_path)
    shot_data = _load_or_create_shots(
        filtered_circuit, dem, shots_path, BIVARIATE_NUM_SHOTS
    )
    mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    priors = mats.priors
    shots = shot_data[:, 0 : dem.num_detectors]
    observables = shot_data[:, dem.num_detectors :]
    return {
        "mats": mats,
        "priors": priors,
        "shots": shots,
        "observables": observables,
    }


def test_bbcode_ilp_decoder_end_to_end(gurobi_env, bbcode_data):
    mats = bbcode_data["mats"]
    priors = bbcode_data["priors"]
    shots = bbcode_data["shots"]
    observables = bbcode_data["observables"]

    decoder = _build_ilp_decoder(
        gurobi_env, mats.check_matrix, mats.observables_matrix, priors
    )
    ilp_results = decoder.decode_batch_result(shots)
    assert all(result.success for result in ilp_results)

    predicted_observables = np.stack(
        [result.predicted_observables for result in ilp_results], axis=0
    )
    assert predicted_observables.shape == observables.shape

    predicted_errors = np.stack([result.error_vector for result in ilp_results], axis=0)
    logical_errors = is_logical_error(
        observables, mats.observables_matrix, predicted_errors
    )
    logical_errors = np.asarray(logical_errors)
    assert logical_errors.shape[0] == shots.shape[0]


def test_bbcode_ilp_decoder_logical_gap_end_to_end(gurobi_env, bbcode_data):
    mats = bbcode_data["mats"]
    priors = bbcode_data["priors"]
    shots = bbcode_data["shots"]

    decoder = _build_ilp_decoder(
        gurobi_env, mats.check_matrix, mats.observables_matrix, priors
    )
    ilp_results = decoder.decode_batch_result(
        shots,
        get_logicalgap=True,
        logical_gap_flip_last_detector=False,
    )
    ilp_gap_values = [result.metadata.get("logical_gap") for result in ilp_results]
    print("ILP logical gap values:", ilp_gap_values)
    assert all(gap is not None for gap in ilp_gap_values)
    ilp_gaps = np.array([float(gap) for gap in ilp_gap_values], dtype=float)
    assert ilp_gaps.shape[0] == shots.shape[0]