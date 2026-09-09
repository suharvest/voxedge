"""RK ASR admits more sessions than it runs inferences.

``supports_parallel=False`` (only one inference at a time on the single shared
RKNNLite context) must hold no matter what. ``max_concurrent`` — the session
admission ceiling the host reads — rises only for inner backends whose
inference is confined to ``finalize()``.

The allow-list is the load-bearing part: a streaming inner backend also infers
inside ``get_partial()``, which the host calls on the event-loop thread
outside its per-utterance slot. Admitting extra sessions there would put two
inferences on one context at the same time.
"""

import pytest

from voxedge.backends.rk.asr import (
    RKASRBackend,
    RKASRConfig,
    _resolve_max_sessions,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("ASR_BACKEND", "ASR_MAX_SESSIONS"):
        monkeypatch.delenv(key, raising=False)


def _profile(**env):
    return {"name": "rk3576-sensevoice", "env": env}


# ── in-flight inference never moves ────────────────────────────────────


def test_inference_is_never_declared_parallel():
    """No profile, env or override may make the shared context parallel."""
    for profile in (
        None,
        _profile(ASR_BACKEND="sensevoice_rknn"),
        _profile(ASR_BACKEND="sensevoice_rknn", ASR_MAX_SESSIONS="16"),
        _profile(ASR_BACKEND="paraformer_rknn"),
    ):
        cap = RKASRBackend.concurrency_capability(profile)
        assert cap.supports_parallel is False, (
            "the shared RKNNLite context can only run one inference at a time"
        )
        assert cap.requires_exclusive_device is True


# ── session capacity ───────────────────────────────────────────────────


def test_sensevoice_gets_the_default_session_capacity():
    cap = RKASRBackend.concurrency_capability(
        _profile(ASR_BACKEND="sensevoice_rknn")
    )
    assert cap.max_concurrent == RKASRConfig.max_sessions
    assert cap.max_concurrent > 1


def test_streaming_inner_backends_stay_single_session():
    """They infer in get_partial(), outside the host's per-utterance slot."""
    for inner in ("paraformer_rknn", "qwen3_rk", "sensevoice_sherpa", ""):
        cap = RKASRBackend.concurrency_capability(_profile(ASR_BACKEND=inner))
        assert cap.max_concurrent == 1, f"{inner!r} must not fan out sessions"


def test_unknown_inner_backend_defaults_to_single_session():
    """Allow-list, not deny-list: a new backend is conservative until checked."""
    cap = RKASRBackend.concurrency_capability(
        _profile(ASR_BACKEND="some_future_rknn_backend")
    )
    assert cap.max_concurrent == 1


# ── configuration ──────────────────────────────────────────────────────


def test_profile_env_overrides_the_default():
    cap = RKASRBackend.concurrency_capability(
        _profile(ASR_BACKEND="sensevoice_rknn", ASR_MAX_SESSIONS="6")
    )
    assert cap.max_concurrent == 6


def test_profile_env_wins_over_process_env(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "sensevoice_rknn")
    monkeypatch.setenv("ASR_MAX_SESSIONS", "2")
    assert _resolve_max_sessions(_profile(
        ASR_BACKEND="sensevoice_rknn", ASR_MAX_SESSIONS="7",
    )) == 7


def test_process_env_is_used_when_no_profile_is_passed(monkeypatch):
    """The host's capability probe calls the classmethod without a profile.

    It exports the profile's env block into os.environ before backends are
    imported, so this is the path production actually takes.
    """
    monkeypatch.setenv("ASR_BACKEND", "sensevoice_rknn")
    monkeypatch.setenv("ASR_MAX_SESSIONS", "5")
    assert RKASRBackend.concurrency_capability().max_concurrent == 5


def test_no_profile_and_no_env_is_conservative():
    assert _resolve_max_sessions(None) == 1


def test_garbage_and_out_of_range_values_do_not_crash(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "sensevoice_rknn")
    assert _resolve_max_sessions(
        _profile(ASR_BACKEND="sensevoice_rknn", ASR_MAX_SESSIONS="not-a-number")
    ) == RKASRConfig.max_sessions
    assert _resolve_max_sessions(
        _profile(ASR_BACKEND="sensevoice_rknn", ASR_MAX_SESSIONS="0")
    ) == 1
    assert _resolve_max_sessions(
        _profile(ASR_BACKEND="sensevoice_rknn", ASR_MAX_SESSIONS="")
    ) == RKASRConfig.max_sessions


def test_scaling_mode_unchanged():
    cap = RKASRBackend.concurrency_capability(
        _profile(ASR_BACKEND="sensevoice_rknn")
    )
    assert cap.scaling_mode == "external_managed"
    assert cap.is_stateful is True
