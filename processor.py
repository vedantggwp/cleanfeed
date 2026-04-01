import logging

import torch
import torchaudio

from engine import PodcastEngine


logger = logging.getLogger(__name__)

_ENGINE: PodcastEngine | None = None


def _get_engine() -> PodcastEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PodcastEngine()
    return _ENGINE


def shutdown_engine() -> None:
    global _ENGINE
    if _ENGINE is not None:
        del _ENGINE
        _ENGINE = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("Engine unloaded and MPS cache cleared.")


def process_audio(input_path: str, output_path: str) -> None:
    logger.info("Loading audio from %s", input_path)
    wav, sample_rate = torchaudio.load(input_path)

    # Convert to mono by averaging channels
    mono = wav.mean(dim=0).flatten()

    if mono.numel() == 0:
        raise ValueError("Input audio is empty.")

    logger.info(
        "Processing %d samples at %d Hz (%.1fs)",
        mono.numel(),
        sample_rate,
        mono.numel() / sample_rate,
    )

    engine = _get_engine()
    enhanced, output_sr = engine.enhance(mono, sample_rate)

    if enhanced.ndim != 1:
        raise ValueError("Engine output must be a 1D mono waveform tensor.")

    enhanced = enhanced.detach().cpu()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    logger.info("Saving enhanced audio to %s", output_path)
    torchaudio.save(output_path, enhanced.unsqueeze(0), output_sr)
