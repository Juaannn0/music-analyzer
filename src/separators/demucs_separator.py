"""
Demucs v4 Source Separation backend implementation.
Extracts vocals and computes instrumental by summing non-vocal stems.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch

from src.separators.base import BaseSeparator
from src.logger import get_logger

class DemucsSeparator(BaseSeparator):
    """Demucs v4 separation engine with automatic instrumental derivation."""

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cpu",
        shifts: int = 1,
        overlap: float = 0.25,
        segment_length: Optional[float] = 7.8
    ):
        self.model_name = model_name
        self.device = device
        self.shifts = max(1, shifts)
        self.overlap = max(0.01, min(0.99, overlap))
        self.segment_length = segment_length
        self.model = None
        self.logger = get_logger()

    def load_model(self) -> None:
        """Loads pretrained Demucs model."""
        if self.model is not None:
            return

        from demucs.pretrained import get_model
        self.logger.debug(f"Cargando modelo Demucs '{self.model_name}' en dispositivo '{self.device}'...")
        self.model = get_model(name=self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def separate(self, audio_path: Path) -> Dict[str, Any]:
        """
        Executes Demucs forward pass on the audio file.
        Derives vocals.wav and instrumental.wav from the predicted stems.
        """
        self.load_model()
        from demucs.apply import apply_model
        from demucs.audio import AudioFile

        reader = AudioFile(str(audio_path))
        wav = reader.read(
            streams=0,
            samplerate=self.model.samplerate,
            channels=self.model.audio_channels
        )

        sources = list(self.model.sources)
        if "vocals" not in sources:
            raise ValueError(f"El modelo Demucs '{self.model_name}' no incluye 'vocals' en sus stems: {sources}")

        vocals_idx = sources.index("vocals")
        non_vocals_indices = [i for i, s in enumerate(sources) if s != "vocals"]

        wav_input = wav[None].to(self.device)

        try:
            with torch.no_grad():
                out = apply_model(
                    self.model,
                    wav_input,
                    shifts=self.shifts,
                    split=True,
                    overlap=self.overlap,
                    segment=self.segment_length,
                    device=self.device
                )
        except torch.cuda.OutOfMemoryError:
            self.logger.warning("CUDA Out of Memory durante separación. Reintentando en CPU...")
            torch.cuda.empty_cache()
            self.model.to("cpu")
            self.device = "cpu"
            with torch.no_grad():
                out = apply_model(
                    self.model,
                    wav[None].to("cpu"),
                    shifts=1,
                    split=True,
                    overlap=self.overlap,
                    segment=self.segment_length,
                    device="cpu"
                )

        stems = out[0]
        vocals = stems[vocals_idx].detach().cpu()
        instrumental = sum(stems[i] for i in non_vocals_indices).detach().cpu()

        return {
            "vocals": vocals,
            "instrumental": instrumental,
            "sample_rate": self.model.samplerate
        }
