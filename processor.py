import logging

import torch
import torchaudio

from engine import ResembleEngine


logger = logging.getLogger(__name__)

_ENGINE: ResembleEngine | None = None


def _get_engine() -> ResembleEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ResembleEngine()
    return _ENGINE


def shutdown_engine() -> None:
    global _ENGINE
    if _ENGINE is not None:
        del _ENGINE
        _ENGINE = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("Engine unloaded and MPS cache cleared.")


def _chunk_starts(total_samples: int, chunk_samples: int, step: int) -> list[int]:
    if total_samples <= chunk_samples:
        return [0]

    return list(range(0, total_samples, step))


def _crossfade_chunks(previous: torch.Tensor, current: torch.Tensor, overlap_samples: int) -> torch.Tensor:
    overlap_len = min(overlap_samples, previous.numel(), current.numel())
    if overlap_len <= 0:
        return torch.cat([previous, current], dim=0)

    if overlap_len == 1:
        blended = 0.5 * previous[-1:] + 0.5 * current[:1]
    else:
        # Use the rising half of a Hann window as the fade curve.
        hann = torch.hann_window(
            overlap_len * 2,
            periodic=False,
            dtype=previous.dtype,
            device=previous.device,
        )[:overlap_len]
        blended = previous[-overlap_len:] * (1 - hann) + current[:overlap_len] * hann

    return torch.cat([previous[:-overlap_len], blended, current[overlap_len:]], dim=0)


def process_audio(input_path: str, output_path: str) -> None:
    logger.info("Loading audio from %s", input_path)
    wav, sample_rate = torchaudio.load(input_path)
    wav = wav.mean(dim=0)
    wav = wav.flatten()

    if wav.numel() == 0:
        raise ValueError("Input audio is empty.")

    chunk_samples = sample_rate * 10
    overlap_samples = sample_rate * 1
    step = chunk_samples - overlap_samples

    starts = _chunk_starts(wav.numel(), chunk_samples, step)
    logger.info(
        "Processing %d samples at %d Hz across %d chunk(s)",
        wav.numel(),
        sample_rate,
        len(starts),
    )

    engine = _get_engine()
    enhanced_chunks: list[torch.Tensor] = []
    output_sr: int | None = None
    output_overlap_samples: int | None = None

    for index, start in enumerate(starts, start=1):
        end = min(start + chunk_samples, wav.numel())
        chunk = wav[start:end]
        logger.info("Enhancing chunk %d/%d", index, len(starts))

        enhanced_chunk, chunk_output_sr = engine.enhance(chunk, sample_rate)
        if enhanced_chunk.ndim != 1:
            raise ValueError("Engine output must be a 1D mono waveform tensor.")

        enhanced_chunk = enhanced_chunk.detach().cpu()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if output_sr is None:
            output_sr = chunk_output_sr
            output_overlap_samples = max(1, round(overlap_samples * output_sr / sample_rate))
        elif chunk_output_sr != output_sr:
            raise ValueError("Engine returned inconsistent sample rates across chunks.")

        enhanced_chunks.append(enhanced_chunk)

    result = enhanced_chunks[0]
    for chunk in enhanced_chunks[1:]:
        result = _crossfade_chunks(result, chunk, output_overlap_samples or 0)

    logger.info("Saving enhanced audio to %s", output_path)
    torchaudio.save(output_path, result.unsqueeze(0), output_sr)
    del enhanced_chunks
    del result
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
