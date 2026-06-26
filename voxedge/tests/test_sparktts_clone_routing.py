"""SparkTTS clone voice-registry + request-routing unit tests (P3).

No worker / engines required — exercises VoiceRegistry loading and the
controllable↔clone routing in SparkTTSBackend._build_request.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from voxedge.backends.jetson.sparktts_trt import SparkTTSBackend, SparkTTSConfig
from voxedge.backends.jetson.voice_registry import (
    VoiceRegistry,
    VoiceProfile,
    load_voice_profile,
)


def _write_profile(d, voice_id, *, with_ref_semantic=False):
    safe = voice_id.replace(":", "_").replace("/", "_")
    g = list(range(32))
    ref_sem = [100, 101, 102] if with_ref_semantic else []
    np.savez(
        str(d / f"{safe}.npz"),
        global_ids=np.array(g, dtype=np.int32),
        ref_semantic_ids=np.array(ref_sem, dtype=np.int32),
        d_vector=np.zeros(1024, dtype=np.float32),
    )
    j = {
        "voice_id": voice_id,
        "ref_text": "参考转写" if with_ref_semantic else None,
        "sample_rate": 16000,
        "npz_file": f"{safe}.npz",
    }
    (d / f"{safe}.json").write_text(json.dumps(j, ensure_ascii=False))
    return safe


def test_load_profile_roundtrip(tmp_path):
    _write_profile(tmp_path, "clone:alice", with_ref_semantic=True)
    prof = load_voice_profile(str(tmp_path / "clone_alice.json"))
    assert prof.voice_id == "clone:alice"
    assert len(prof.global_ids) == 32
    assert prof.ref_semantic_ids == [100, 101, 102]
    assert prof.ref_text == "参考转写"


def test_load_profile_rejects_wrong_global_count(tmp_path):
    np.savez(str(tmp_path / "bad.npz"),
             global_ids=np.arange(10, dtype=np.int32),
             ref_semantic_ids=np.array([], dtype=np.int32),
             d_vector=np.zeros(1024, dtype=np.float32))
    (tmp_path / "bad.json").write_text(json.dumps({"voice_id": "bad", "npz_file": "bad.npz"}))
    with pytest.raises(ValueError):
        load_voice_profile(str(tmp_path / "bad.json"))


def test_registry_scan_and_reload(tmp_path):
    _write_profile(tmp_path, "clone:alice")
    reg = VoiceRegistry(str(tmp_path))
    assert reg.contains("clone:alice")
    assert not reg.contains("clone:bob")
    assert {v["voice_id"] for v in reg.list_voices()} == {"clone:alice"}
    # add a second voice → reload picks it up
    _write_profile(tmp_path, "clone:bob")
    assert reg.reload() == 2
    assert reg.contains("clone:bob")


def test_registry_skips_corrupt_profile(tmp_path):
    _write_profile(tmp_path, "clone:good")
    (tmp_path / "broken.json").write_text("{not valid json")
    reg = VoiceRegistry(str(tmp_path))
    assert reg.contains("clone:good")
    assert len(reg.list_voices()) == 1  # broken one skipped, not fatal


def test_worker_request_fields_strategy_a_vs_b(tmp_path):
    _write_profile(tmp_path, "clone:b", with_ref_semantic=True)
    prof = load_voice_profile(str(tmp_path / "clone_b.json"))
    a = prof.worker_request_fields(use_ref_semantic=False)
    assert a["mode"] == "clone" and len(a["global_ids"]) == 32
    assert "ref_semantic_ids" not in a  # strategy A: global only
    b = prof.worker_request_fields(use_ref_semantic=True)
    assert b["ref_semantic_ids"] == [100, 101, 102]
    assert b["ref_text"] == "参考转写"


def _backend(tmp_path, **cfg_kw):
    cfg = SparkTTSConfig(voices_dir=str(tmp_path), **cfg_kw)
    return SparkTTSBackend(cfg)


def test_build_request_routes_clone_on_registry_hit(tmp_path):
    _write_profile(tmp_path, "clone:alice")
    be = _backend(tmp_path)
    req = be._build_request("r1", "你好", stream=True, kwargs={"voice": "clone:alice"})
    assert req["mode"] == "clone"
    assert len(req["global_ids"]) == 32
    assert "gender" not in req  # clone does not send controllable style


def test_build_request_falls_back_to_controllable_on_miss(tmp_path):
    _write_profile(tmp_path, "clone:alice")
    be = _backend(tmp_path)
    # a speaker spec that is NOT a registered voice → controllable style parsing
    req = be._build_request("r2", "hi", stream=True, kwargs={"speaker": "male_high_moderate"})
    assert "mode" not in req
    assert req["gender"] == "male" and req["pitch"] == "high"


def test_build_request_clone_strategy_b_opt_in(tmp_path):
    _write_profile(tmp_path, "clone:b", with_ref_semantic=True)
    be = _backend(tmp_path, clone_use_ref_semantic=True)
    req = be._build_request("r3", "hi", stream=True, kwargs={"voice_id": "clone:b"})
    assert req["mode"] == "clone"
    assert req["ref_semantic_ids"] == [100, 101, 102]


def test_clone_capability_advertised_only_with_voices_dir(tmp_path):
    from voxedge.backends.base import TTSCapability
    with_dir = _backend(tmp_path)
    assert TTSCapability.VOICE_CLONE in with_dir.capabilities
    without = SparkTTSBackend(SparkTTSConfig())
    assert TTSCapability.VOICE_CLONE not in without.capabilities
