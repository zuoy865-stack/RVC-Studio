"""验证说话/唱歌模式的默认值和完整透传。"""

from __future__ import annotations

import tempfile
import unittest
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from infer.module.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from infer.vc.modules import VC
from infer.vc.pipeline import Pipeline
from infer.vc.singing import (
    SINGING_NOISE_SCALE,
    correct_octave_errors,
    normalize_inference_mode,
)
from rvc_core.engine import RVCEngine


class InferenceModeTests(unittest.TestCase):
    def test_mode_normalization_defaults_to_speech(self):
        self.assertEqual(normalize_inference_mode(None), "speech")
        self.assertEqual(normalize_inference_mode(""), "speech")
        self.assertEqual(normalize_inference_mode("speech"), "speech")
        self.assertEqual(normalize_inference_mode("singing"), "singing")
        with self.assertRaises(ValueError):
            normalize_inference_mode("unknown")

    def test_public_inference_layers_default_to_speech(self):
        targets = (VC.vc_single, VC.vc_multi, Pipeline.pipeline, Pipeline.vc)
        for target in targets:
            self.assertEqual(signature(target).parameters["mode"].default, "speech")

    def test_generator_default_noise_stays_original(self):
        targets = (SynthesizerTrnMs256NSFsid.infer, SynthesizerTrnMs256NSFsid_nono.infer)
        for target in targets:
            self.assertEqual(signature(target).parameters["noise_scale"].default, 0.66666)

    def test_singing_octave_correction_keeps_stable_note(self):
        values = np.array([220.0, 220.0, 440.0, 220.0, 220.0])
        corrected = correct_octave_errors(values, np.full(5, 0.2))
        self.assertAlmostEqual(corrected[2], 220.0)

    def test_rmvpe_confidence_is_requested_only_for_singing(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.window = 160
        pipeline.sr = 16000
        pipeline.device = "cpu"
        pipeline.is_half = False
        rmvpe = Mock()

        def infer(audio, thred=0.03, return_confidence=False):
            f0 = np.full(10, 220.0)
            if return_confidence:
                return f0, np.full(10, 0.9)
            return f0

        rmvpe.infer_from_audio.side_effect = infer
        pipeline.model_rmvpe = rmvpe

        with patch("infer.vc.pipeline.process_singing_f0", wraps=lambda f0, confidence: f0) as process:
            pipeline.get_f0(np.zeros(1600), 10, 0, "rmvpe")
            self.assertNotIn("return_confidence", rmvpe.infer_from_audio.call_args.kwargs)
            process.assert_not_called()

            pipeline.get_f0(np.zeros(1600), 10, 0, "rmvpe", mode="singing")
            self.assertTrue(rmvpe.infer_from_audio.call_args.kwargs["return_confidence"])
            process.assert_called_once()

    def test_engine_convert_forwards_default_and_singing_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "voice.pth"
            input_path = root / "input.wav"
            model_path.touch()
            input_path.touch()

            engine = RVCEngine.__new__(RVCEngine)
            engine.model_path = model_path.resolve()
            engine.config = SimpleNamespace(device="mps")
            engine.vc = SimpleNamespace(
                net_g=object(),
                hubert_model=object(),
                pipeline=SimpleNamespace(),
                vc_single=Mock(return_value=("ok", (16000, np.zeros(160)))),
            )
            base_request = {
                "model": str(model_path),
                "input": str(input_path),
                "output": str(root / "output.wav"),
                "overwrite": True,
            }

            with (
                patch.object(RVCEngine, "inspect_model", return_value={}),
                patch.object(RVCEngine, "asset_status", return_value={"hubert": True, "rmvpe": True}),
                patch.object(RVCEngine, "_write_audio"),
                patch("rvc_core.engine.require_mps_module"),
                patch("rvc_core.engine.synchronize"),
                patch("rvc_core.engine.memory_snapshot", return_value={}),
            ):
                engine.convert(dict(base_request))
                self.assertEqual(engine.vc.vc_single.call_args.kwargs["mode"], "speech")

                singing_request = dict(base_request, mode="singing")
                engine.convert(singing_request)
                self.assertEqual(engine.vc.vc_single.call_args.kwargs["mode"], "singing")

    def test_engine_batch_keeps_mode_for_each_child_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            input_dir.mkdir()
            (input_dir / "a.wav").touch()
            (input_dir / "b.flac").touch()
            engine = RVCEngine.__new__(RVCEngine)
            engine.convert = Mock(side_effect=lambda request: {"output": request["output"]})

            result = RVCEngine.convert_batch(
                engine,
                {
                    "input": str(input_dir),
                    "output_dir": str(root / "outputs"),
                    "format": "wav",
                    "mode": "singing",
                },
            )

            self.assertEqual(result["total"], 2)
            self.assertEqual(engine.convert.call_count, 2)
            for call in engine.convert.call_args_list:
                self.assertEqual(call.args[0]["mode"], "singing")

    def test_vc_multi_forwards_mode_to_vc_single(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.wav"
            source.touch()
            vc = VC.__new__(VC)
            vc.vc_single = Mock(return_value=("ok", (16000, np.zeros(160))))

            list(
                vc.vc_multi(
                    0,
                    "",
                    str(root / "outputs"),
                    [str(source)],
                    0,
                    "rmvpe",
                    "",
                    0.75,
                    0,
                    1.0,
                    0.33,
                    "wav",
                    mode="singing",
                )
            )

            self.assertEqual(vc.vc_single.call_args.kwargs["mode"], "singing")

    def test_pipeline_passes_mode_to_f0_and_segment_inference(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.window = 160
        pipeline.t_max = 10**9
        pipeline.t_pad = 160
        pipeline.t_pad2 = 320
        pipeline.t_pad_tgt = 1
        pipeline.device = "cpu"
        pipeline.get_f0 = Mock(
            return_value=(np.full(22, 100, dtype=np.int32), np.full(22, 220, dtype=np.float32))
        )
        pipeline.vc = Mock(return_value=np.zeros(640, dtype=np.float32))

        pipeline.pipeline(
            object(),
            object(),
            0,
            np.zeros(3200, dtype=np.float32),
            [0, 0, 0],
            0,
            "rmvpe",
            "",
            0.0,
            1,
            40000,
            0,
            1.0,
            "v2",
            0.33,
            mode="singing",
        )

        self.assertEqual(pipeline.get_f0.call_args.kwargs["mode"], "singing")
        self.assertEqual(pipeline.vc.call_args.kwargs["mode"], "singing")

    def test_pipeline_vc_uses_original_and_singing_noise_values(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.is_half = False
        pipeline.device = "cpu"
        pipeline.window = 160
        calls = []

        class Generator:
            def infer(self, *args, **kwargs):
                calls.append(kwargs)
                return (torch.zeros(1, 1, 160),)

        def run_graph(module, key, function, *args):
            return function(*args)

        common = (
            object(),
            Generator(),
            torch.zeros(1, dtype=torch.long),
            np.zeros(1600, dtype=np.float32),
            torch.full((1, 10), 100, dtype=torch.long),
            torch.full((1, 10), 220.0),
            [0, 0, 0],
            None,
            None,
            0.0,
            "v2",
            0.5,
        )

        with (
            patch("infer.vc.pipeline.extract_hubert_features", return_value=torch.zeros(1, 10, 4)),
            patch("infer.vc.pipeline.run_cuda_graph", side_effect=run_graph),
            patch("infer.vc.pipeline.require_mps_tensor"),
            patch("infer.vc.pipeline.synchronize"),
        ):
            pipeline.vc(*common)
            pipeline.vc(*common, mode="singing")

        self.assertNotIn("noise_scale", calls[0])
        self.assertEqual(calls[1]["noise_scale"], SINGING_NOISE_SCALE)

    def test_retrieval_temporal_smoothing_is_singing_only(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.is_half = False
        pipeline.device = "cpu"
        pipeline.window = 160

        class Index:
            def search(self, values, k):
                return np.ones((len(values), k)), np.zeros((len(values), k), dtype=np.int64)

        class Generator:
            def infer(self, *args, **kwargs):
                return (torch.zeros(1, 1, 160),)

        def run_graph(module, key, function, *args):
            return function(*args)

        common = (
            object(),
            Generator(),
            torch.zeros(1, dtype=torch.long),
            np.zeros(1600, dtype=np.float32),
            torch.full((1, 10), 100, dtype=torch.long),
            torch.full((1, 10), 220.0),
            [0, 0, 0],
            Index(),
            np.zeros((1, 4), dtype=np.float32),
            0.75,
            "v2",
            0.5,
        )

        with (
            patch("infer.vc.pipeline.extract_hubert_features", return_value=torch.zeros(1, 10, 4)),
            patch("infer.vc.pipeline.run_cuda_graph", side_effect=run_graph),
            patch("infer.vc.pipeline.require_mps_tensor"),
            patch("infer.vc.pipeline.synchronize"),
            patch("infer.vc.pipeline.smooth_retrieval_features", side_effect=lambda values, f0: values) as smooth,
        ):
            pipeline.vc(*common)
            smooth.assert_not_called()
            pipeline.vc(*common, mode="singing")
            smooth.assert_called_once()


if __name__ == "__main__":
    unittest.main()
