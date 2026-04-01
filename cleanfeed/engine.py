"""5-stage audio enhancement pipeline.

DeepFilterNet3 → MossFormer2 → Pedalboard DSP → LUFS → Limiter.
Pure tensor/numpy in, tensor out. Zero filesystem I/O.
"""

import logging

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from clearvoice import ClearVoice
from df.enhance import enhance as df_enhance
from df.enhance import init_df
from pedalboard import (
    Compressor,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    PeakFilter,
    Pedalboard,
)

logger = logging.getLogger(__name__)

OUTPUT_SR = 48000
LUFS_TARGET = -18.0
LIMITER_CEILING_DB = -1.5


class Engine:
    """5-stage pipeline: DeepFilterNet → MossFormer2 → Pedalboard DSP → LUFS → Limiter."""

    def __init__(self) -> None:
        logger.info("Loading DeepFilterNet3...")
        self._df_model, self._df_state, _ = init_df()
        self._df_sr = self._df_state.sr()
        logger.info("DeepFilterNet3 loaded (sr=%d)", self._df_sr)

        logger.info("Loading MossFormer2_SE_48K...")
        self._clearvoice = ClearVoice(
            task="speech_enhancement",
            model_names=["MossFormer2_SE_48K"],
        )
        logger.info("MossFormer2_SE_48K loaded")

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
        self._limiter = Pedalboard([Limiter(threshold_db=LIMITER_CEILING_DB)])
        logger.info("Mastering chain ready")

    def enhance(self, audio_tensor: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        """Run the full enhancement pipeline.

        Args:
            audio_tensor: 1D mono float32 tensor.
            sample_rate: Input sample rate.

        Returns:
            (enhanced_tensor, output_sample_rate) — 1D mono tensor at 48kHz.
        """
        if audio_tensor.ndim != 1:
            raise ValueError("audio_tensor must be 1D mono")

        audio_tensor = audio_tensor.detach().flatten().to(dtype=torch.float32).cpu().contiguous()
        logger.info(
            "Starting enhancement: %d samples at %d Hz (%.1fs)",
            audio_tensor.numel(),
            sample_rate,
            audio_tensor.numel() / sample_rate,
        )

        # --- Stage 1: DeepFilterNet3 noise suppression ---
        if sample_rate != self._df_sr:
            logger.info("Resampling %d → %d Hz for DeepFilterNet", sample_rate, self._df_sr)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, self._df_sr)

        denoised = df_enhance(self._df_model, self._df_state, audio_tensor.unsqueeze(0))
        denoised_1d = denoised.squeeze(0).detach().cpu().contiguous()
        logger.info("Stage 1 complete: DeepFilterNet denoise")

        # --- Stage 2: MossFormer2 speech enhancement ---
        cv_input = denoised_1d.numpy().astype(np.float32)[np.newaxis, :]
        with torch.no_grad():
            cv_result = self._clearvoice(cv_input)

        if isinstance(cv_result, dict):
            enhanced_np = list(cv_result.values())[0]
        elif isinstance(cv_result, torch.Tensor):
            enhanced_np = cv_result.numpy()
        else:
            enhanced_np = np.asarray(cv_result, dtype=np.float32)

        enhanced_np = np.ascontiguousarray(enhanced_np.astype(np.float32, copy=False))
        if enhanced_np.ndim == 1:
            enhanced_np = enhanced_np[np.newaxis, :]
        logger.info("Stage 2 complete: MossFormer2 enhance")

        # --- Stage 3: Pedalboard DSP mastering ---
        mastered = self._mastering(enhanced_np, sample_rate=OUTPUT_SR, reset=True)
        logger.info("Stage 3 complete: DSP mastering")

        # --- Stage 4: LUFS normalization ---
        mono = np.ascontiguousarray(mastered[0].astype(np.float64, copy=False))
        meter = pyln.Meter(OUTPUT_SR)
        loudness = meter.integrated_loudness(mono)
        if np.isfinite(loudness):
            normalized = pyln.normalize.loudness(mono, loudness, LUFS_TARGET).astype(np.float32)
            logger.info("Stage 4 complete: LUFS %.1f → %.1f", loudness, LUFS_TARGET)
        else:
            logger.warning("LUFS non-finite (%.2f), skipping normalization", loudness)
            normalized = mono.astype(np.float32)

        # --- Stage 5: Brick-wall limiter ---
        limited = self._limiter(normalized[np.newaxis, :], sample_rate=OUTPUT_SR, reset=True)
        logger.info("Stage 5 complete: Limiter at %.1f dB", LIMITER_CEILING_DB)

        result = torch.from_numpy(np.ascontiguousarray(limited[0]))
        return result, OUTPUT_SR
