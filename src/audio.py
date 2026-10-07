"""
Audio I/O, metadata extraction via Mutagen, and optional DSP acoustic preprocessing.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf
from mutagen import File as MutagenFile

def extract_metadata(audio_path: Path, output_json_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Extracts metadata from audio file using mutagen.
    Saves to metadata.json if output_json_path is provided.
    Falls back gracefully if tags are missing.
    """
    default_title = audio_path.stem
    meta: Dict[str, Any] = {
        "title": default_title,
        "artist": None,
        "album": None,
        "track_number": None,
        "genre": None,
        "year": None,
        "duration_seconds": None,
        "channels": None,
        "sample_rate": None,
        "source_filename": audio_path.name
    }

    try:
        tag_file = MutagenFile(str(audio_path))
        if tag_file is not None:
            if tag_file.info:
                meta["duration_seconds"] = round(getattr(tag_file.info, "length", 0.0), 2)
                meta["channels"] = getattr(tag_file.info, "channels", None)
                meta["sample_rate"] = getattr(tag_file.info, "sample_rate", None)

            tags = getattr(tag_file, "tags", None) or tag_file
            if tags:
                # Helper to extract first tag value
                def get_tag_val(*keys):
                    for k in keys:
                        if k in tags:
                            val = tags[k]
                            if isinstance(val, (list, tuple)) and len(val) > 0:
                                return str(val[0])
                            return str(val)
                    return None

                # ID3, Vorbis, MP4 tag keys
                title = get_tag_val("TIT2", "title", "\xa9nam")
                if title:
                    meta["title"] = title

                artist = get_tag_val("TPE1", "artist", "\xa9ART")
                if artist:
                    meta["artist"] = artist

                album = get_tag_val("TALB", "album", "\xa9alb")
                if album:
                    meta["album"] = album

                track = get_tag_val("TRCK", "tracknumber", "trkn")
                if track:
                    meta["track_number"] = track.split("/")[0] if "/" in track else track

                genre = get_tag_val("TCON", "genre", "\xa9gen")
                if genre:
                    meta["genre"] = genre

                year = get_tag_val("TDRC", "date", "\xa9day", "TYER")
                if year:
                    meta["year"] = str(year)[:4]
    except Exception:
        # Never fail pipeline because of metadata reading error
        pass

    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta

def save_wav(tensor_or_array, output_path: Path, sample_rate: int = 44100) -> None:
    """
    Saves audio to WAV PCM 16-bit. Accepts torch.Tensor [channels, samples]
    or numpy ndarray [samples, channels] / [channels, samples].
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If torch Tensor
    try:
        import torch
        if isinstance(tensor_or_array, torch.Tensor):
            data = tensor_or_array.detach().cpu().numpy()
            # If shape is [channels, samples], transpose to [samples, channels] for soundfile
            if data.ndim == 2 and data.shape[0] < data.shape[1]:
                data = data.T
        else:
            data = np.asarray(tensor_or_array)
    except ImportError:
        data = np.asarray(tensor_or_array)

    # Clip to avoid overflow distortion before writing
    data = np.clip(data, -1.0, 1.0)
    sf.write(str(output_path), data, sample_rate, subtype="PCM_16")

def preprocess_vocals_for_transcription(
    input_wav: Path,
    output_wav: Path,
    normalize_peak: bool = True,
    high_pass_filter: bool = False
) -> Path:
    """
    Preprocesses separated vocal stem for optimal Whisper transcription.
    Applies optional -1.0 dBFS peak normalization and 80Hz high-pass filter.
    Returns path to preprocessed audio.
    """
    if not normalize_peak and not high_pass_filter:
        return input_wav

    try:
        data, sr = sf.read(str(input_wav))
        
        # High-pass filter at 80 Hz to reduce sub-bass Demucs phase artifacts
        if high_pass_filter:
            from scipy.signal import butter, sosfilt
            sos = butter(2, 80, btype="highpass", fs=sr, output="sos")
            if data.ndim == 1:
                data = sosfilt(sos, data)
            else:
                for ch in range(data.shape[1]):
                    data[:, ch] = sosfilt(sos, data[:, ch])

        # Peak normalization to -1.0 dBFS
        if normalize_peak:
            peak = np.max(np.abs(data))
            if peak > 1e-5:
                target_peak = 10.0 ** (-1.0 / 20.0)  # ~0.891
                data = data * (target_peak / peak)

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), data, sr, subtype="PCM_16")
        return output_wav
    except Exception:
        # Fall back to original file if preprocessing encounters any issue
        return input_wav
