import types
import sys
import torchaudio

if not hasattr(torchaudio, "backend"):
    backend_module = types.ModuleType("torchaudio.backend")
    common_module = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:
        def __init__(
            self,
            sample_rate: int = 0,
            num_frames: int = 0,
            num_channels: int = 0,
            bits_per_sample: int = 0,
            encoding: str = "",
        ) -> None:
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    common_module.AudioMetaData = AudioMetaData
    backend_module.common = common_module
    sys.modules["torchaudio.backend"] = backend_module
    sys.modules["torchaudio.backend.common"] = common_module
    torchaudio.backend = backend_module

import logging

import numpy as np
import onnxruntime as ort
import pyloudnorm as pyln
import torch
from df.enhance import enhance as df_enhance
from df.enhance import init_df
from huggingface_hub import hf_hub_download
from pedalboard import (
    Compressor,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    PeakFilter,
    Pedalboard,
)


logger = logging.getLogger(__name__)


class PodcastEngine:
    def __init__(self) -> None:
        logger.info("Loading DeepFilterNet3...")
        self._df_model, self._df_state, _ = init_df()
        self._df_sr = self._df_state.sr()
        logger.info("DeepFilterNet3 loaded (sr=%d)", self._df_sr)

        logger.info("Loading FlashSR ONNX model...")
        model_path = hf_hub_download(
            repo_id="YatharthS/FlashSR",
            filename="model.onnx",
            subfolder="onnx",
        )
        self._sr_session = ort.InferenceSession(model_path)
        logger.info("FlashSR loaded")

        self._mastering = Pedalboard(
            [
                HighpassFilter(cutoff_frequency_hz=80),
                PeakFilter(cutoff_frequency_hz=300, gain_db=-3.0, q=1.0),
                Compressor(threshold_db=-20, ratio=2.0, attack_ms=15, release_ms=100),
                Compressor(threshold_db=-15, ratio=3.0, attack_ms=10, release_ms=80),
                PeakFilter(cutoff_frequency_hz=6000, gain_db=-4.0, q=2.0),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=2.5, q=0.8),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0, q=0.7),
            ]
        )
        self._limiter = Pedalboard([Limiter(threshold_db=-1.5)])
        logger.info("Mastering chain ready")

    def enhance(self, audio_tensor: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        if audio_tensor.ndim != 1:
            raise ValueError("audio_tensor must be 1D mono")

        audio_tensor = audio_tensor.detach().flatten().to(dtype=torch.float32).cpu().contiguous()
        logger.info(
            "Starting enhancement for %d samples at %d Hz",
            audio_tensor.numel(),
            sample_rate,
        )

        if sample_rate != self._df_sr:
            logger.info("Resampling input from %d Hz to %d Hz for DeepFilterNet", sample_rate, self._df_sr)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, self._df_sr)

        audio_2d = audio_tensor.unsqueeze(0)
        denoised = df_enhance(self._df_model, self._df_state, audio_2d)
        logger.info("Stage 1 complete: DeepFilterNet denoise")

        denoised_16k = torchaudio.functional.resample(
            denoised.squeeze(0).detach().cpu().contiguous(),
            self._df_sr,
            16000,
        )
        sr_input = denoised_16k.numpy().astype(np.float32, copy=False)[np.newaxis, :]
        sr_output = self._sr_session.run(["reconstruction"], {"audio_values": sr_input})[0]
        audio_np = np.ascontiguousarray(sr_output.astype(np.float32, copy=False))
        logger.info("Stage 2 complete: FlashSR super-resolution")

        mastered = self._mastering(audio_np, sample_rate=48000, reset=True)
        logger.info("Stage 3 complete: DSP mastering")

        mono = np.ascontiguousarray(mastered[0].astype(np.float64, copy=False))
        meter = pyln.Meter(48000)
        loudness = meter.integrated_loudness(mono)
        if np.isfinite(loudness):
            normalized = pyln.normalize.loudness(mono, loudness, -16.0).astype(np.float32)
        else:
            logger.warning("LUFS non-finite, skipping normalization")
            normalized = mono.astype(np.float32)

        limited = self._limiter(normalized[np.newaxis, :], sample_rate=48000, reset=True)
        logger.info("Stage 4 complete: LUFS normalization + limiter")

        result = torch.from_numpy(np.ascontiguousarray(limited[0]))
        return result, 48000


ResembleEngine = PodcastEngine


def shutdown_engine() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("Engine shutdown complete.")
