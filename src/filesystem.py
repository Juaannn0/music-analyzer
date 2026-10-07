"""
Filesystem utilities, audio file scanning, Windows path sanitization and resume logic.
"""

import re
from pathlib import Path
from typing import List, Dict

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
INVALID_WIN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

def scan_audio_files(input_dir: Path) -> List[Path]:
    """Scans the directory for supported audio files (case-insensitive)."""
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    
    files = []
    for item in input_dir.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(item)
    return sorted(files, key=lambda p: p.name.lower())

def sanitize_folder_name(name: str) -> str:
    """Sanitizes track name for Windows and cross-platform directory safety."""
    cleaned = INVALID_WIN_CHARS.sub("_", name)
    cleaned = cleaned.strip(". ")
    if not cleaned:
        cleaned = "unnamed_track"
    return cleaned[:120]

def get_track_directories(output_root: Path, track_stem: str) -> Dict[str, Path]:
    """Calculates and creates directory paths for a track."""
    folder_name = sanitize_folder_name(track_stem)
    track_dir = output_root / folder_name
    vocals_dir = track_dir / "vocals"
    instrumental_dir = track_dir / "instrumental"

    vocals_dir.mkdir(parents=True, exist_ok=True)
    instrumental_dir.mkdir(parents=True, exist_ok=True)

    return {
        "track_dir": track_dir,
        "vocals_dir": vocals_dir,
        "instrumental_dir": instrumental_dir,
        "vocals_wav": vocals_dir / "vocals.wav",
        "instrumental_wav": instrumental_dir / "instrumental.wav",
        "transcription_txt": vocals_dir / "transcription.txt",
        "transcription_srt": vocals_dir / "transcription.srt",
        "transcription_vtt": vocals_dir / "transcription.vtt",
        "metadata_json": track_dir / "metadata.json",
    }

def has_valid_separation(paths: Dict[str, Path]) -> bool:
    """Checks if both vocals.wav and instrumental.wav exist and are non-empty (>1KB)."""
    vw = paths["vocals_wav"]
    iw = paths["instrumental_wav"]
    return (vw.is_file() and vw.stat().st_size > 1024 and
            iw.is_file() and iw.stat().st_size > 1024)

def has_valid_transcription(paths: Dict[str, Path]) -> bool:
    """Checks if all transcription artifacts (.txt, .srt, .vtt) exist."""
    txt = paths["transcription_txt"]
    srt = paths["transcription_srt"]
    vtt = paths["transcription_vtt"]
    return txt.is_file() and srt.is_file() and vtt.is_file()
