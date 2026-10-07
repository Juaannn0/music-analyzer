"""
Base abstraction for source separation backends.
Allows future backends (e.g., RoFormer, MDX-Net) to plug in without altering the pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

class BaseSeparator(ABC):
    """Abstract interface for audio source separation."""

    @abstractmethod
    def load_model(self) -> None:
        """Loads and initializes the separation model."""
        pass

    @abstractmethod
    def separate(self, audio_path: Path) -> Dict[str, Any]:
        """
        Processes an audio file and returns a dictionary with:
        {
            'vocals': tensor_or_array,
            'instrumental': tensor_or_array,
            'sample_rate': int
        }
        """
        pass
