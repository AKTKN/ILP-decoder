import os
import sys

import numpy as np
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ilp_decoder import DecoderConfig, ILPDecoder
from ilp_decoder.core import DecoderDependencies
from ilp_decoder.formulation import BuildModelResult
from ilp_decoder.models import DecodeResult


class _FakeVar:
    def __init__(self, value: int = 0) -> None:
        self.X = value


class _FakeConstr:
    def __init__(self) -> None:
        self.RHS = 0


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.end_calls = 0

    def end(self) -> None:
        self.end_calls += 1


class _FakeSolverBackend:
    def solve(self, model_result, syndrome, *, config):
        syndrome = np.asarray(syndrome, dtype=np.int64)
        if model_result.model.name == "gap":
            return DecodeResult(
                success=True,
                error_vector=np.array([0, 1, 0], dtype=np.int64),
                objective_value=5.0,
                runtime_ms=0.0,
                status="OPTIMAL",
                metadata={"solver": "fake"},
            )
        return DecodeResult(
            success=True,
            error_vector=np.array([1, 0, 0], dtype=np.int64),
            objective_value=3.0,
            runtime_ms=0.0,
            status="OPTIMAL",
            metadata={"solver": "fake", "syndrome_sum": int(syndrome.sum())},
        )


def _make_builder(model_store, name: str):
    def _builder(env, parity_check_matrix, observables, config, **kwargs):
        model = _FakeModel(name)
        model_store.append(model)
        auxiliary_vars = {"parity_slack": [_FakeVar(0)]}
        if name == "gap":
            auxiliary_vars["logical_xor"] = [_FakeVar(1)]
        return BuildModelResult(
            model=model,
            error_vars=[_FakeVar(0) for _ in range(parity_check_matrix.shape[1])],
            auxiliary_vars=auxiliary_vars,
            syndrome_constraints=[_FakeConstr() for _ in range(parity_check_matrix.shape[0])],
        )

    return _builder


def _build_decoder(base_models, gap_models):
    parity_check_matrix = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int64)
    observables = np.array([[1, 1, 0]], dtype=np.int64)
    prior = np.array([0.1, 0.1, 0.1], dtype=float)
    deps = DecoderDependencies(
        env=object(),
        model_builder=_make_builder(base_models, "base"),
        gap_model_builder=_make_builder(gap_models, "gap"),
        solver_backend=_FakeSolverBackend(),
    )
    return ILPDecoder(
        parity_check_matrix=parity_check_matrix,
        observables=observables,
        prior=prior,
        config=DecoderConfig(log_to_console=False),
        deps=deps,
    )


def test_gap_detail_includes_stage_weights_and_stage2_solution():
    base_models = []
    gap_models = []
    decoder = _build_decoder(base_models, gap_models)

    result = decoder.decode_result(
        np.array([1, 0], dtype=np.int64),
        get_logicalgap=True,
        get_gap_detail=True,
    )

    assert result.metadata["logical_gap"] == pytest.approx(2.0)
    assert result.metadata["obs_flip_idx"] == [0]
    assert np.array_equal(result.error_vector, np.array([1, 0, 0], dtype=np.int64))

    gap_detail = result.metadata["gap_detail"]
    assert gap_detail["stage1_weight"] == pytest.approx(3.0)
    assert gap_detail["stage2_weight"] == pytest.approx(5.0)
    assert np.array_equal(
        gap_detail["stage2_error_vector"],
        np.array([0, 1, 0], dtype=np.int64),
    )

    decoder.close()


def test_get_gap_detail_requires_logical_gap():
    base_models = []
    gap_models = []
    decoder = _build_decoder(base_models, gap_models)

    with pytest.raises(ValueError, match="get_gap_detail requires get_logicalgap=True"):
        decoder.decode_result(np.array([1, 0], dtype=np.int64), get_gap_detail=True)

    decoder.close()


def test_decoder_releases_owned_models():
    base_models = []
    gap_models = []
    decoder = _build_decoder(base_models, gap_models)

    override_config = DecoderConfig(log_to_console=False, threads=1)
    result = decoder.decode_result(
        np.array([1, 1], dtype=np.int64),
        config=override_config,
        get_logical_gap=True,
        get_gap_detail=True,
    )

    assert result.success
    assert len(base_models) == 2
    assert len(gap_models) == 1
    assert base_models[0].end_calls == 0
    assert base_models[1].end_calls == 1
    assert gap_models[0].end_calls == 1

    decoder.close()
    assert base_models[0].end_calls == 1

    decoder.close()
    assert base_models[0].end_calls == 1

    with pytest.raises(RuntimeError, match="ILPDecoder has been closed"):
        decoder.decode_result(np.array([1, 0], dtype=np.int64))
