"""Tests for voxedge.backends.jetson._deploy_paths and build_config_from_env().

Covers:
  1. Env-override: explicit env var → config field uses that value (bypasses resolver)
  2. No-env fallback: no relevant env vars → resolvers return non-crashing values
  3. Shim import: server.core.deploy_paths re-exports from _deploy_paths (needs
     seeed-local-voice on sys.path — skipped when not importable)
"""

import importlib
import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_deploy_paths(monkeypatch):
    """Reload _deploy_paths so module-level constants re-evaluate the patched env."""
    monkeypatch.delitem(sys.modules, "voxedge.backends.jetson._deploy_paths", raising=False)
    import voxedge.backends.jetson._deploy_paths as dp
    return importlib.reload(dp)


def _clear_all_path_env(monkeypatch):
    """Remove all env vars that _deploy_paths reads, to test empty-string fallback."""
    for key in [
        "EDGE_LLM_BASE", "EDGE_LLM_BUILD_DIR",
        "OVS_BASE", "OVS_WORKER_BUILD",
        "EDGE_LLM_TTS_BIN", "EDGE_LLM_TTS_WORKER_BIN",
        "EDGE_LLM_ASR_BIN", "EDGE_LLM_ASR_WORKER_BIN",
        "EDGELLM_PLUGIN_PATH", "EDGE_LLM_ASR_PLUGIN_PATH", "EDGELLM_ASR_PLUGIN_PATH",
        "EDGE_LLM_TTS_TALKER_DIR", "EDGE_LLM_TTS_FULL_TALKER_DIR",
        "EDGE_LLM_TTS_PRUNED_TALKER_DIR", "EDGE_LLM_TTS_VOCAB_PRUNED",
        "QWEN3_TTS_VOCAB_PRUNED", "EDGE_LLM_TTS_CP_DIR", "EDGE_LLM_TTS_CP_BF16_IO_DIR",
        "EDGE_LLM_TTS_CODE2WAV_DIR", "EDGE_LLM_TTS_TOKENIZER_DIR",
        "EDGE_LLM_ASR_FULL_ENGINE_DIR", "EDGE_LLM_ASR_PRUNED_ENGINE_DIR",
        "EDGE_LLM_ASR_ENGINE_DIR", "EDGE_LLM_ASR_VOCAB_PRUNED",
        "EDGE_LLM_ASR_AUDIO_ENC_DIR",
        "EDGE_LLM_QWEN3_PROFILE", "OVS_QWEN3_PROFILE",
    ]:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# 1. Env-override tests: build_config_from_env() uses explicit env vars
# ---------------------------------------------------------------------------

class TestTTSBuildConfigFromEnvOverrides:
    def setup_method(self):
        # Lazy import avoids heavy CUDA deps at collection time.
        from voxedge.backends.jetson.trt_edge_llm_tts import build_config_from_env
        self.build = build_config_from_env

    def test_tts_binary_override(self):
        env = {"EDGE_LLM_TTS_BIN": "/custom/tts_binary"}
        cfg = self.build(env=env)
        assert cfg.tts_binary == "/custom/tts_binary"

    def test_worker_binary_override(self):
        env = {"EDGE_LLM_TTS_WORKER_BIN": "/custom/worker"}
        cfg = self.build(env=env)
        assert cfg.worker_binary == "/custom/worker"

    def test_plugin_path_override(self):
        env = {"EDGELLM_PLUGIN_PATH": "/custom/plugin.so"}
        cfg = self.build(env=env)
        assert cfg.plugin_path == "/custom/plugin.so"

    def test_talker_dir_override(self):
        env = {"EDGE_LLM_TTS_TALKER_DIR": "/custom/talker"}
        cfg = self.build(env=env)
        assert cfg.talker_dir == "/custom/talker"

    def test_code_predictor_dir_override(self):
        env = {"EDGE_LLM_TTS_CP_DIR": "/custom/cp"}
        cfg = self.build(env=env)
        assert cfg.code_predictor_dir == "/custom/cp"

    def test_tokenizer_dir_override(self):
        env = {"EDGE_LLM_TTS_TOKENIZER_DIR": "/custom/tok"}
        cfg = self.build(env=env)
        assert cfg.tokenizer_dir == "/custom/tok"

    def test_code2wav_dir_override(self):
        env = {"EDGE_LLM_TTS_CODE2WAV_DIR": "/custom/c2w"}
        cfg = self.build(env=env)
        assert cfg.code2wav_dir == "/custom/c2w"

    def test_model_id_override(self):
        env = {"OVS_TTS_MODEL_ID": "my_tts_model"}
        cfg = self.build(env=env)
        assert cfg.model_id == "my_tts_model"

    def test_backend_mode_override(self):
        env = {"OVS_TTS_BACKEND": "direct"}
        cfg = self.build(env=env)
        assert cfg.backend_mode == "direct"

    def test_worker_concurrency_override(self):
        env = {"OVS_TTS_WORKER_CONCURRENCY": "4"}
        cfg = self.build(env=env)
        assert cfg.worker_concurrency == 4

    def test_sampling_overrides(self):
        env = {
            "OVS_TTS_TALKER_TEMPERATURE": "0.5",
            "OVS_TTS_TALKER_TOP_K": "20",
            "OVS_TTS_TOP_P": "0.8",
            "OVS_TTS_SEED": "123",
            "TTS_MAX_AUDIO_LENGTH": "512",
            "TTS_MIN_AUDIO_LENGTH": "10",
            "TTS_REPETITION_PENALTY": "1.1",
        }
        cfg = self.build(env=env)
        assert cfg.talker_temperature == 0.5
        assert cfg.talker_top_k == 20
        assert cfg.talker_top_p == 0.8
        assert cfg.seed == 123
        assert cfg.max_audio_length == 512
        assert cfg.min_audio_length == 10
        assert cfg.repetition_penalty == pytest.approx(1.1)

    def test_segmentation_overrides(self):
        env = {
            "EDGE_LLM_TTS_SEGMENT_TEXT": "0",
            "EDGE_LLM_TTS_SEGMENT_MAX_CHARS": "80",
            "EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS": "32",
            "EDGE_LLM_TTS_SEGMENT_PAUSE_MS": "60",
            "EDGE_LLM_TTS_HARD_SEGMENT_PAUSE_MS": "100",
        }
        cfg = self.build(env=env)
        assert cfg.segment_text is False
        assert cfg.segment_max_chars_latin == 80
        assert cfg.segment_max_chars_cjk == 32
        assert cfg.segment_pause_ms == 60
        assert cfg.segment_hard_pause_ms == 100

    def test_streaming_overrides(self):
        env = {
            "EDGE_LLM_TTS_STREAMING_PROFILE": "low_latency",
            "EDGE_LLM_TTS_FIRST_CHUNK_FRAMES": "64",
            "EDGE_LLM_TTS_CHUNK_FRAMES": "128",
            "EDGE_LLM_TTS_ADAPTIVE_CHUNKS": "1",
            "EDGE_LLM_TTS_MAX_CHUNK_FRAMES": "512",
            "EDGE_LLM_TTS_CHUNK_GROWTH_FRAMES": "32",
        }
        cfg = self.build(env=env)
        assert cfg.streaming_profile == "low_latency"
        assert cfg.first_chunk_frames == 64
        assert cfg.chunk_frames == 128
        assert cfg.adaptive_chunks is True
        assert cfg.max_chunk_frames == 512
        assert cfg.chunk_growth_frames == 32

    def test_stateful_code2wav_explicit_false(self):
        env = {"EDGE_LLM_TTS_STATEFUL_CODE2WAV": "0"}
        cfg = self.build(env=env)
        assert cfg.stateful_code2wav is False

    def test_stateful_code2wav_explicit_true(self):
        env = {"EDGE_LLM_TTS_STATEFUL_CODE2WAV": "1"}
        cfg = self.build(env=env)
        assert cfg.stateful_code2wav is True

    def test_stateful_code2wav_unset_is_none(self):
        env = {}
        cfg = self.build(env=env)
        assert cfg.stateful_code2wav is None


class TestASRBuildConfigFromEnvOverrides:
    def setup_method(self):
        from voxedge.backends.jetson.trt_edge_llm_asr import build_config_from_env
        self.build = build_config_from_env

    def test_asr_binary_override(self):
        env = {"EDGE_LLM_ASR_BIN": "/custom/asr_bin"}
        cfg = self.build(env=env)
        assert cfg.asr_binary == "/custom/asr_bin"

    def test_worker_binary_override(self):
        env = {"EDGE_LLM_ASR_WORKER_BIN": "/custom/asr_worker"}
        cfg = self.build(env=env)
        assert cfg.worker_binary == "/custom/asr_worker"

    def test_plugin_path_override(self):
        env = {"EDGE_LLM_ASR_PLUGIN_PATH": "/custom/asr_plugin.so"}
        cfg = self.build(env=env)
        assert cfg.plugin_path == "/custom/asr_plugin.so"

    def test_engine_dir_override(self):
        env = {"EDGE_LLM_ASR_ENGINE_DIR": "/custom/asr_engine"}
        cfg = self.build(env=env)
        assert cfg.engine_dir == "/custom/asr_engine"

    def test_audio_encoder_dir_override(self):
        env = {"EDGE_LLM_ASR_AUDIO_ENC_DIR": "/custom/audio_enc"}
        cfg = self.build(env=env)
        assert cfg.audio_encoder_dir == "/custom/audio_enc"

    def test_max_slots_override(self):
        env = {"EDGE_LLM_ASR_MAX_CONCURRENT": "3"}
        cfg = self.build(env=env)
        assert cfg.max_slots == 3

    def test_sampling_overrides(self):
        env = {
            "ASR_TEMPERATURE": "0.7",
            "ASR_TOP_P": "0.9",
            "ASR_TOP_K": "5",
            "ASR_MAX_GENERATE_LENGTH": "100",
        }
        cfg = self.build(env=env)
        assert cfg.temperature == pytest.approx(0.7)
        assert cfg.top_p == pytest.approx(0.9)
        assert cfg.top_k == 5
        assert cfg.max_generate_length == 100

    def test_stream_mode_override(self):
        env = {"EDGE_LLM_ASR_STREAM_MODE": "token_by_token"}
        cfg = self.build(env=env)
        assert cfg.stream_mode == "token_by_token"

    def test_warmup_skip_env(self):
        env = {"SKIP_ASR_WARMUP": "1"}
        cfg = self.build(env=env)
        assert cfg.worker_warmup is False

    def test_warmup_worker_warmup_disabled(self):
        env = {"EDGE_LLM_ASR_WORKER_WARMUP": "0"}
        cfg = self.build(env=env)
        assert cfg.worker_warmup is False

    def test_cuda_graph_override(self):
        env = {"EDGE_LLM_ASR_CUDA_GRAPH": "1"}
        cfg = self.build(env=env)
        assert cfg.worker_cuda_graph == "1"

    def test_offline_segment_disabled(self):
        env = {"EDGE_LLM_ASR_OFFLINE_SEGMENT": "0"}
        cfg = self.build(env=env)
        assert cfg.offline_segment_enabled is False


# ---------------------------------------------------------------------------
# 2. No-env fallback: resolvers should not crash when env is empty
# ---------------------------------------------------------------------------

class TestDeployPathsNoEnvFallback:
    def test_resolver_no_crash_empty_env(self, monkeypatch):
        """All resolvers must return a string (possibly empty) without crashing."""
        _clear_all_path_env(monkeypatch)
        dp = _reload_deploy_paths(monkeypatch)

        # These may return non-empty strings (~/... paths) but must not raise.
        assert isinstance(dp.resolve_tts_worker_binary(), str)
        assert isinstance(dp.resolve_asr_worker_binary(), str)
        assert isinstance(dp.resolve_tts_talker_dir(), str)
        assert isinstance(dp.resolve_tts_code_predictor_dir(), str)
        assert isinstance(dp.resolve_tts_tokenizer_dir(), str)
        assert isinstance(dp.resolve_tts_code2wav_dir(), str)
        assert isinstance(dp.resolve_plugin_path(), str)

    def test_tts_build_config_no_crash_empty_env(self):
        """build_config_from_env({}) must not raise."""
        from voxedge.backends.jetson.trt_edge_llm_tts import build_config_from_env
        cfg = build_config_from_env(env={})
        assert cfg is not None
        # Path fields may be non-empty (resolved from ~/... defaults), but must be str.
        assert isinstance(cfg.tts_binary, str)
        assert isinstance(cfg.worker_binary, str)
        assert isinstance(cfg.talker_dir, str)

    def test_asr_build_config_no_crash_empty_env(self):
        """build_config_from_env({}) must not raise."""
        from voxedge.backends.jetson.trt_edge_llm_asr import build_config_from_env
        cfg = build_config_from_env(env={})
        assert cfg is not None
        assert isinstance(cfg.asr_binary, str)
        assert isinstance(cfg.worker_binary, str)
        assert isinstance(cfg.engine_dir, str)

    def test_ovs_worker_build_empty_no_crash(self, monkeypatch):
        """When OVS_BASE and OVS_WORKER_BUILD are unset, worker binary resolvers
        fall back to the edgellm build path (no crash, no ~/project/openvoicestream)."""
        monkeypatch.delenv("OVS_BASE", raising=False)
        monkeypatch.delenv("OVS_WORKER_BUILD", raising=False)
        monkeypatch.delenv("EDGE_LLM_TTS_WORKER_BIN", raising=False)
        monkeypatch.delenv("EDGE_LLM_ASR_WORKER_BIN", raising=False)
        dp = _reload_deploy_paths(monkeypatch)
        tts_w = dp.resolve_tts_worker_binary()
        asr_w = dp.resolve_asr_worker_binary()
        assert isinstance(tts_w, str)
        assert isinstance(asr_w, str)
        # Must NOT contain the old hardcode
        assert "openvoicestream" not in tts_w
        assert "openvoicestream" not in asr_w

    def test_ovs_worker_build_env_used(self, monkeypatch, tmp_path):
        """OVS_WORKER_BUILD env var is used when set."""
        worker_dir = tmp_path / "workers"
        worker_dir.mkdir()
        tts_bin = worker_dir / "qwen3_tts_worker"
        tts_bin.touch()
        monkeypatch.setenv("OVS_WORKER_BUILD", str(worker_dir))
        monkeypatch.delenv("EDGE_LLM_TTS_WORKER_BIN", raising=False)
        dp = _reload_deploy_paths(monkeypatch)
        assert dp.resolve_tts_worker_binary() == str(tts_bin)


# ---------------------------------------------------------------------------
# 4. env=None path: build_config_from_env() reads os.environ by default
# ---------------------------------------------------------------------------

def test_tts_build_config_env_none_reads_os_environ(monkeypatch):
    """env=None 时应读 os.environ（生产默认路径）。"""
    monkeypatch.setenv("EDGE_LLM_TTS_WORKER_BIN", "/tmp/fake_tts_worker")
    from voxedge.backends.jetson.trt_edge_llm_tts import build_config_from_env
    cfg = build_config_from_env()  # env=None
    assert cfg.worker_binary == "/tmp/fake_tts_worker"


def test_asr_build_config_env_none_reads_os_environ(monkeypatch):
    """env=None 时应读 os.environ（生产默认路径）。"""
    monkeypatch.setenv("EDGE_LLM_ASR_WORKER_BIN", "/tmp/fake_asr_worker")
    from voxedge.backends.jetson.trt_edge_llm_asr import build_config_from_env
    cfg = build_config_from_env()  # env=None
    assert cfg.worker_binary == "/tmp/fake_asr_worker"


# ---------------------------------------------------------------------------
# 3. Shim backward-compat: server.core.deploy_paths re-exports from voxedge
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    "server.core.deploy_paths" not in sys.modules
    and not any("seeed-local-voice" in p for p in sys.path),
    reason="seeed-local-voice not on sys.path",
)
def test_shim_import_resolve_tts_worker_binary():
    """server.core.deploy_paths.resolve_tts_worker_binary is importable via shim."""
    try:
        import importlib
        import sys
        # Remove cached shim so we get a clean import
        for mod in list(sys.modules.keys()):
            if "server.core.deploy_paths" in mod or "voxedge.backends.jetson._deploy_paths" in mod:
                del sys.modules[mod]
        from server.core.deploy_paths import resolve_tts_worker_binary, TTS_BINARY
        assert callable(resolve_tts_worker_binary)
        assert isinstance(TTS_BINARY, str)
    except ImportError:
        pytest.skip("server package not importable from this venv")
