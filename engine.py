import logging
import os
from functools import cache

import numpy as np
import pyloudnorm as pyln
import torch
from pedalboard import Compressor, HighpassFilter, HighShelfFilter, Limiter, PeakFilter, Pedalboard


logger = logging.getLogger(__name__)


@cache
def _get_denoise():
    os.environ.setdefault("DS_ACCELERATOR", "cpu")
    from resemble_enhance.denoiser.inference import denoise as denoise_fn

    return denoise_fn


class PodcastEngine:
    def __init__(self) -> None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device: str = "mps"
        else:
            self.device = "cpu"
            logger.warning("MPS unavailable or PyTorch not built with MPS support; falling back to CPU.")

        logger.info("Using device: %s", self.device)

        self._mastering_board = Pedalboard(
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
        self._limiter_board = Pedalboard([Limiter(threshold_db=-1.5)])

    def denoise_only(self, audio_tensor: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        audio_np, output_sr = self._run_denoise(audio_tensor, sample_rate)
        return self._to_output_tensor(audio_np), output_sr

    def enhance(self, audio_tensor: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        audio_np, output_sr = self._run_denoise(audio_tensor, sample_rate)

        mastered = self._mastering_board(
            self._to_pedalboard_input(audio_np),
            sample_rate=output_sr,
            reset=True,
        )

        normalized = self._normalize_loudness(mastered, output_sr)
        limited = self._limiter_board(normalized, sample_rate=output_sr, reset=True)

        return self._to_output_tensor(limited), output_sr

    def _run_denoise(self, audio_tensor: torch.Tensor, sample_rate: int) -> tuple[np.ndarray, int]:
        if audio_tensor.ndim != 1:
            raise ValueError("audio_tensor must be a 1D mono waveform tensor.")

        logger.debug("Running denoise-only stage on %d samples at %d Hz.", audio_tensor.numel(), sample_rate)

        input_tensor = audio_tensor.detach().flatten().to(dtype=torch.float32).cpu().contiguous()
        result_tensor, output_sr = _get_denoise()(
            input_tensor,
            sample_rate,
            run_dir=None,
            device=self.device,
        )
        result_tensor = result_tensor.detach().cpu().flatten().to(dtype=torch.float32).contiguous()
        return result_tensor.numpy(), output_sr

    def _normalize_loudness(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        audio_for_lufs = self._to_loudness_input(audio_np)
        meter = pyln.Meter(sample_rate)

        try:
            loudness = meter.integrated_loudness(audio_for_lufs)
        except ValueError as exc:
            logger.warning("LUFS measurement failed; skipping normalization: %s", exc)
            return self._to_pedalboard_input(audio_np)

        if not np.isfinite(loudness):
            logger.warning("LUFS measurement returned a non-finite value; skipping normalization.")
            return self._to_pedalboard_input(audio_np)

        normalized = pyln.normalize.loudness(audio_for_lufs, loudness, -16.0)
        return self._from_loudness_output(normalized)

    @staticmethod
    def _to_pedalboard_input(audio_np: np.ndarray) -> np.ndarray:
        audio_np = np.asarray(audio_np, dtype=np.float32)
        if audio_np.ndim == 1:
            return np.ascontiguousarray(audio_np[np.newaxis, :])
        if audio_np.ndim == 2:
            return np.ascontiguousarray(audio_np)
        raise ValueError("Audio must be a 1D or 2D numpy array.")

    @staticmethod
    def _to_loudness_input(audio_np: np.ndarray) -> np.ndarray:
        audio_np = np.asarray(audio_np, dtype=np.float64)
        if audio_np.ndim == 1:
            return np.ascontiguousarray(audio_np)
        if audio_np.ndim == 2:
            return np.ascontiguousarray(audio_np.T)
        raise ValueError("Audio must be a 1D or 2D numpy array.")

    @staticmethod
    def _from_loudness_output(audio_np: np.ndarray) -> np.ndarray:
        audio_np = np.asarray(audio_np, dtype=np.float32)
        if audio_np.ndim == 1:
            return np.ascontiguousarray(audio_np[np.newaxis, :])
        if audio_np.ndim == 2:
            return np.ascontiguousarray(audio_np.T)
        raise ValueError("Audio must be a 1D or 2D numpy array.")

    @staticmethod
    def _to_output_tensor(audio_np: np.ndarray) -> torch.Tensor:
        audio_np = np.asarray(audio_np, dtype=np.float32)
        if audio_np.ndim == 2:
            if audio_np.shape[0] != 1:
                raise ValueError("Engine output must be mono.")
            audio_np = audio_np[0]
        elif audio_np.ndim != 1:
            raise ValueError("Engine output must be a 1D mono waveform.")

        return torch.from_numpy(np.ascontiguousarray(audio_np))


ResembleEngine = PodcastEngine


def shutdown_engine() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("Engine shutdown complete.")
