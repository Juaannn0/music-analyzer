"""
Faster-Whisper Transcription engine and subtitle/text formatters.
Generates TXT with [MM:SS.xx], SubRip (.srt) and WebVTT (.vtt) files.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import soundfile as sf
from src.logger import get_logger

def load_audio_for_whisper(audio_path: Path) -> np.ndarray:
    """
    Loads audio using soundfile and converts to 16kHz mono float32 ndarray.
    Bypasses PyAV container decoding issues and ensures clean input.
    """
    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        import math
        gcd = math.gcd(16000, sr)
        up = 16000 // gcd
        down = sr // gcd
        data = resample_poly(data, up, down).astype(np.float32)
    return data

def format_timestamp_txt(seconds: float) -> str:
    """Formats seconds as [MM:SS.xx]."""
    total_sec = max(0.0, seconds)
    minutes = int(total_sec // 60)
    rem_sec = total_sec % 60
    return f"[{minutes:02d}:{rem_sec:05.2f}]"

def format_timestamp_srt(seconds: float) -> str:
    """Formats seconds as HH:MM:SS,mmm for SubRip."""
    total_sec = max(0.0, seconds)
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    rem_sec = total_sec % 60
    secs = int(rem_sec)
    millis = int(round((rem_sec - secs) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_timestamp_vtt(seconds: float) -> str:
    """Formats seconds as HH:MM:SS.mmm for WebVTT."""
    total_sec = max(0.0, seconds)
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    rem_sec = total_sec % 60
    secs = int(rem_sec)
    millis = int(round((rem_sec - secs) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def format_txt(segments: List[Dict[str, Any]]) -> str:
    """Produces TXT format with timestamp prefix on each line."""
    lines = []
    for s in segments:
        text = s.get("text", "").strip()
        if text:
            ts = format_timestamp_txt(s.get("start", 0.0))
            lines.append(f"{ts} {text}")
    return "\n".join(lines) + ("\n" if lines else "")

def format_srt(segments: List[Dict[str, Any]]) -> str:
    """Produces valid SubRip (.srt) subtitle format."""
    blocks = []
    idx = 1
    for s in segments:
        text = s.get("text", "").strip()
        if text:
            start_str = format_timestamp_srt(s.get("start", 0.0))
            end_str = format_timestamp_srt(s.get("end", 0.0))
            blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
            idx += 1
    return "\n".join(blocks)

def format_vtt(segments: List[Dict[str, Any]]) -> str:
    """Produces valid WebVTT (.vtt) format."""
    lines = ["WEBVTT\n"]
    for s in segments:
        text = s.get("text", "").strip()
        if text:
            start_str = format_timestamp_vtt(s.get("start", 0.0))
            end_str = format_timestamp_vtt(s.get("end", 0.0))
            lines.append(f"{start_str} --> {end_str}\n{text}\n")
    return "\n".join(lines)

class WhisperTranscriber:
    """Faster-Whisper transcription engine with Silero VAD."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        vad: bool = True,
        condition_on_previous_text: bool = False,
        beam_size: int = 5,
        temperature: float = 0.0
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad = vad
        self.condition_on_previous_text = condition_on_previous_text
        self.beam_size = beam_size
        self.temperature = temperature
        self.model = None
        self.logger = get_logger()

    def load_model(self) -> None:
        """Initializes WhisperModel if not already loaded."""
        if self.model is not None:
            return

        from faster_whisper import WhisperModel
        self.logger.debug(
            f"Cargando modelo Faster-Whisper '{self.model_size}' "
            f"(device={self.device}, compute_type={self.compute_type})..."
        )
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    def transcribe(self, audio_path: Path, initial_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribes vocal audio using Faster-Whisper and returns segments.
        Loads audio into memory to ensure stability and compatibility.
        """
        self.load_model()
        audio_array = load_audio_for_whisper(audio_path)

        vad_params = dict(min_silence_duration_ms=800, speech_pad_ms=400) if self.vad else None

        segments_gen, info = self.model.transcribe(
            audio_array,
            language=self.language,
            task="transcribe",
            beam_size=self.beam_size,
            temperature=self.temperature,
            vad_filter=self.vad,
            vad_parameters=vad_params,
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=initial_prompt,
            word_timestamps=False
        )

        segments = []
        for s in segments_gen:
            segments.append({
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip()
            })

        return {
            "segments": segments,
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 3)
        }

    def save_transcription_outputs(self, result: Dict[str, Any], paths: Dict[str, Path]) -> None:
        """Saves TXT, SRT and VTT files to destination paths."""
        segments = result.get("segments", [])

        txt_content = format_txt(segments)
        srt_content = format_srt(segments)
        vtt_content = format_vtt(segments)

        paths["transcription_txt"].write_text(txt_content, encoding="utf-8")
        paths["transcription_srt"].write_text(srt_content, encoding="utf-8")
        paths["transcription_vtt"].write_text(vtt_content, encoding="utf-8")
