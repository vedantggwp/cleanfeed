"""E2E tests for the public API and CLI."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import soundfile as sf

import cleanfeed


class TestPublicAPI:
    def test_version_exists(self):
        assert hasattr(cleanfeed, "__version__")
        assert cleanfeed.__version__ == "0.1.0"

    def test_exports(self):
        assert callable(cleanfeed.enhance)
        assert callable(cleanfeed.Engine)
        assert callable(cleanfeed.process_audio)
        assert callable(cleanfeed.shutdown_engine)
        assert cleanfeed.OUTPUT_SR == 48000

    @pytest.mark.slow
    def test_enhance_function(self, test_wav_48k):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            result = cleanfeed.enhance(str(test_wav_48k), out_path)
            assert result == out_path
            assert Path(out_path).exists()

            audio, sr = sf.read(out_path, dtype="float32")
            assert sr == 48000
            assert len(audio) > 0
        finally:
            Path(out_path).unlink(missing_ok=True)
            cleanfeed.shutdown_engine()


class TestCLI:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "cleanfeed.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "cleanfeed" in result.stdout
        assert "input" in result.stdout

    def test_missing_input_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "cleanfeed.cli", "nonexistent.wav", "out.wav"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "does not exist" in result.stderr

    def test_unsupported_format(self, tmp_path):
        fake = tmp_path / "test.xyz"
        fake.write_text("not audio")

        result = subprocess.run(
            [sys.executable, "-m", "cleanfeed.cli", str(fake), "out.wav"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "unsupported" in result.stderr.lower()

    @pytest.mark.slow
    def test_cli_e2e(self, test_wav_48k):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "cleanfeed.cli", str(test_wav_48k), out_path],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert Path(out_path).exists()
            assert "Duration:" in result.stdout
            assert "Sample rate:" in result.stdout
        finally:
            Path(out_path).unlink(missing_ok=True)
