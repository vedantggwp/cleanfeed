"""cleanfeed — Local AI audio restoration. Phone recording → podcast quality.

Usage:
    # Simple: file in, file out
    import cleanfeed
    cleanfeed.enhance("recording.m4a", "podcast.wav")

    # Advanced: tensor-level control
    from cleanfeed import Engine
    engine = Engine()
    enhanced, sr = engine.enhance(audio_tensor, sample_rate)
"""

# Torchaudio compatibility shim — MUST be imported before any model code.
from . import _compat as _compat  # noqa: F401

from .engine import Engine, OUTPUT_SR
from .processor import process_audio, shutdown_engine

__version__ = "0.1.0"
__all__ = ["Engine", "enhance", "process_audio", "shutdown_engine", "OUTPUT_SR"]


def enhance(input_path: str, output_path: str) -> str:
    """Enhance an audio file. The simplest way to use cleanfeed.

    Args:
        input_path: Path to input audio file (WAV, M4A, MP3, FLAC, OGG, AAC).
        output_path: Path for enhanced output WAV file.

    Returns:
        The output_path, for chaining.
    """
    process_audio(input_path, output_path)
    return output_path
