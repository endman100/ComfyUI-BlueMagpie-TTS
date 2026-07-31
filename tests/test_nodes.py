import gc
import importlib
import json
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture
def nodes_module(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    registered = {}

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(models_dir)
    folder_paths.add_model_folder_path = lambda name, path, **_kwargs: registered.setdefault(name, []).append(path)
    folder_paths.get_folder_paths = lambda name: registered.get(name, [])
    folder_paths.get_temp_directory = lambda: str(temp_dir)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    sys.modules.pop("nodes", None)
    module = importlib.import_module("nodes")
    module._MODEL_CACHE.clear()
    yield module
    sys.modules.pop("nodes", None)


def _make_model_dir(path: Path, *, include_speaker_table: bool = True):
    path.mkdir(parents=True)
    for name in ("config.json", "pytorch_model.bin", "audiovae.pth", "tokenizer.json"):
        (path / name).write_bytes(b"test")
    if include_speaker_table:
        speaker_table = path / "checkpoints" / "speaker_centroids.pt"
        speaker_table.parent.mkdir(parents=True)
        speaker_table.write_bytes(b"test")


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.linspace(-0.2, 0.2, 16)


def _bundle(nodes_module):
    return nodes_module.BlueMagpieModelBundle(
        model=FakeModel(),
        model_dir="model",
        device="cpu",
        speaker_ids=("hung_yi_lee", "female_voice"),
        speaker_centroids=torch.stack((torch.ones(192), torch.zeros(192))),
        sample_rate=48000,
    )


def test_local_model_resolution_uses_registered_comfy_folder(nodes_module, tmp_path):
    model_dir = tmp_path / "models" / "bluemagpie" / "OpenFormosa" / "BlueMagpie-TTS"
    _make_model_dir(model_dir)

    assert nodes_module._local_model_path("OpenFormosa/BlueMagpie-TTS") == str(model_dir)
    assert nodes_module._model_choices()[0] == "OpenFormosa/BlueMagpie-TTS"


def test_model_directory_requires_built_in_speaker_table(nodes_module, tmp_path):
    model_dir = tmp_path / "incomplete-model"
    _make_model_dir(model_dir, include_speaker_table=False)

    assert not nodes_module._is_model_dir(str(model_dir))


def test_loader_caches_loaded_model(nodes_module, tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "bluemagpie" / "OpenFormosa" / "BlueMagpie-TTS"
    _make_model_dir(model_dir)
    bundle = _bundle(nodes_module)
    calls = []

    def fake_load(path, device):
        calls.append((path, device))
        return bundle

    monkeypatch.setattr(nodes_module, "_load_model_bundle", fake_load)
    loader = nodes_module.BlueMagpieModelLoader()
    first, first_info = loader.load_model("OpenFormosa/BlueMagpie-TTS", "cpu")
    second, _second_info = loader.load_model("OpenFormosa/BlueMagpie-TTS", "cpu")

    assert first is second is bundle
    assert len(calls) == 1
    assert json.loads(first_info)["sample_rate"] == 48000


def test_loader_cache_does_not_pin_model_in_memory(nodes_module, tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "bluemagpie" / "OpenFormosa" / "BlueMagpie-TTS"
    _make_model_dir(model_dir)
    loaded_refs = []

    def fake_load(_path, _device):
        bundle = _bundle(nodes_module)
        loaded_refs.append(weakref.ref(bundle))
        return bundle

    monkeypatch.setattr(nodes_module, "_load_model_bundle", fake_load)
    loaded, _info = nodes_module.BlueMagpieModelLoader().load_model("OpenFormosa/BlueMagpie-TTS", "cpu")

    assert len(nodes_module._MODEL_CACHE) == 1
    del loaded
    gc.collect()
    assert loaded_refs[0]() is None
    assert len(nodes_module._MODEL_CACHE) == 0


def test_auto_device_prefers_mps_when_cuda_is_unavailable(nodes_module, monkeypatch):
    monkeypatch.setattr(nodes_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(nodes_module.torch.backends.mps, "is_available", lambda: True)

    assert nodes_module._resolve_device("auto") == "mps"


def test_tts_uses_built_in_speaker_and_returns_comfy_audio(nodes_module):
    bundle = _bundle(nodes_module)
    audio, info_json = nodes_module.BlueMagpieTTS().generate(
        bundle,
        "這是 test。",
        "hung_yi_lee",
        2.0,
        10,
        2000,
        False,
    )

    assert audio["sample_rate"] == 48000
    assert tuple(audio["waveform"].shape) == (1, 1, 16)
    call = bundle.model.calls[0]
    assert call["target_text"] == "這是 test。"
    assert tuple(call["speaker_centroid"].shape) == (192,)
    assert "reference_wav_path" not in call
    assert json.loads(info_json)["conditioning"] == "built_in:hung_yi_lee"


def test_reference_audio_overrides_speaker_and_temp_file_is_removed(nodes_module):
    bundle = _bundle(nodes_module)
    reference = {"waveform": torch.zeros(1, 1, 4800), "sample_rate": 48000}

    _audio, info_json = nodes_module.BlueMagpieTTS().generate(
        bundle,
        "Clone this voice.",
        "female_voice",
        2.0,
        10,
        2000,
        False,
        reference_audio=reference,
    )

    call = bundle.model.calls[0]
    reference_path = call["reference_wav_path"]
    assert not Path(reference_path).exists()
    assert "speaker_centroid" not in call
    assert json.loads(info_json)["conditioning"] == "reference_audio"


def test_non_finite_reference_audio_is_rejected(nodes_module):
    reference = {"waveform": torch.tensor([[[float("nan")]]]), "sample_rate": 48000}

    with pytest.raises(ValueError, match="non-finite"):
        nodes_module.BlueMagpieTTS().generate(
            _bundle(nodes_module),
            "Clone this voice.",
            "female_voice",
            2.0,
            10,
            2000,
            True,
            reference_audio=reference,
        )


def test_long_reference_path_is_reported_in_generation_info(nodes_module, tmp_path, monkeypatch, caplog):
    reference_path = tmp_path / "long-reference.mp3"
    reference_path.write_bytes(b"test")
    monkeypatch.setattr(nodes_module.sf, "info", lambda _path: types.SimpleNamespace(duration=60.0))

    _audio, info_json = nodes_module.BlueMagpieTTS().generate(
        _bundle(nodes_module),
        "Clone this voice.",
        "female_voice",
        2.0,
        10,
        2000,
        True,
        reference_audio_path=str(reference_path),
    )

    info = json.loads(info_json)
    assert info["reference_duration_seconds"] == 60.0
    assert info["warnings"] == [nodes_module.LONG_REFERENCE_WARNING]
    assert nodes_module.LONG_REFERENCE_WARNING in caplog.text


def test_empty_text_is_rejected(nodes_module):
    with pytest.raises(ValueError, match="Text cannot be empty"):
        nodes_module.BlueMagpieTTS().generate(
            _bundle(nodes_module),
            "  ",
            "hung_yi_lee",
            2.0,
            10,
            2000,
            False,
        )


def test_node_inputs_keep_runtime_surface_small(nodes_module):
    loader_inputs = nodes_module.BlueMagpieModelLoader.INPUT_TYPES()["required"]
    tts_inputs = nodes_module.BlueMagpieTTS.INPUT_TYPES()

    assert list(loader_inputs) == ["model_path", "device"]
    assert "mps" in loader_inputs["device"][0]
    assert tts_inputs["required"]["retry_badcase"][1]["default"] is True
    assert "reference_audio" in tts_inputs["optional"]
    assert "reference_audio_path" in tts_inputs["optional"]
