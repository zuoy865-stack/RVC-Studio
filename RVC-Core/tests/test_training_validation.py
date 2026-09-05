from pathlib import Path

import torch

from train.validation import (
    ValidationTracker,
    ensure_conversion_probes,
    load_probe_audio,
    speaker_similarity_with_leakage_penalty,
    split_train_validation,
    summarize_training_health,
    validation_components,
    weighted_validation_score,
)


class _Dataset:
    def __init__(self):
        self.audiopaths_and_text = [
            ["a.wav"],
            ["a.wav"],
            ["b.wav"],
            ["c.wav"],
            ["d.wav"],
            ["e.wav"],
        ]
        self.lengths = [200] * len(self.audiopaths_and_text)

    def __len__(self):
        return len(self.audiopaths_and_text)

    def __getitem__(self, index):
        return self.audiopaths_and_text[index]


def test_validation_split_is_deterministic_and_keeps_duplicates_together():
    dataset = _Dataset()
    train_a, validation_a = split_train_validation(dataset, 0.2, 1234)
    train_b, validation_b = split_train_validation(dataset, 0.2, 1234)
    assert train_a.indices == train_b.indices
    assert validation_a.indices == validation_b.indices
    train_paths = {dataset[index][0] for index in train_a.indices}
    validation_paths = {dataset[index][0] for index in validation_a.indices}
    assert train_paths.isdisjoint(validation_paths)
    assert train_paths | validation_paths == {"a.wav", "b.wav", "c.wav", "d.wav", "e.wav"}


def test_identical_audio_has_full_weighted_validation_score():
    torch.manual_seed(7)
    mel = torch.randn(2, 80, 48)
    wave = torch.randn(2, 1, 4096)
    components = validation_components(mel, mel, wave, wave, 40000)
    values = {key: float(value) for key, value in components.items()}
    assert all(abs(value - 1.0) < 1e-5 for value in values.values())
    assert abs(weighted_validation_score(values) - 1.0) < 1e-5


def test_tracker_uses_min_delta_for_best_model_and_patience(tmp_path: Path):
    tracker = ValidationTracker(tmp_path / "state.json", patience=2, min_delta=0.002)
    first = tracker.observe(0.5, 10)
    second = tracker.observe(0.501, 20)
    third = tracker.observe(0.499, 30)
    assert first.significant_improvement
    assert not second.is_best and not second.significant_improvement
    assert third.should_stop
    assert tracker.best_score == 0.5
    assert tracker.best_epoch == 10
    assert tracker.bad_count == 2

    tracker.save()
    restored = ValidationTracker(tmp_path / "state.json", patience=2, min_delta=0.002)
    assert restored.best_score == tracker.best_score
    assert restored.best_epoch == tracker.best_epoch
    assert restored.bad_count == tracker.bad_count


def test_source_timbre_leakage_reduces_speaker_score():
    target = torch.tensor([[1.0, 0.0]])
    source = torch.tensor([[0.0, 1.0]])
    converted_target = speaker_similarity_with_leakage_penalty(target, target, source)
    converted_source = speaker_similarity_with_leakage_penalty(source, target, source)
    assert float(converted_target[0]) > float(converted_source[0])
    assert float(converted_target[1]) == 1.0
    assert float(converted_source[2]) == 1.0


def test_unhealthy_gan_loss_rejects_checkpoint(tmp_path: Path):
    healthy = summarize_training_health([25.0, 24.0], [2.0, 2.2])
    collapsed = summarize_training_health([25.0], [0.0])
    assert healthy["healthy"]
    assert not collapsed["healthy"]
    tracker = ValidationTracker(tmp_path / "state.json", patience=2, min_delta=0.002)
    decision = tracker.observe(0.9, 10, checkpoint_eligible=False)
    assert not decision.is_best
    assert tracker.best_epoch == 0


def test_builtin_conversion_probes_cover_four_ranges(tmp_path: Path):
    paths = ensure_conversion_probes(tmp_path)
    probes = load_probe_audio(paths)
    assert [path.name for path in paths] == [
        "builtin_bass.wav",
        "builtin_mid.wav",
        "builtin_high.wav",
        "builtin_sustain.wav",
    ]
    assert len(probes) == 4
    assert max(len(audio) / sample_rate for _, audio, sample_rate in probes) >= 3.0
