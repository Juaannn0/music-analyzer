"""
Configuration management for Music Analyzer V1.
Implements typed dataclasses and merges CLI arguments > config.yaml > defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

@dataclass
class HardwareConfig:
    device: str = "auto"
    num_threads: int = 6

@dataclass
class SeparationConfig:
    backend: str = "demucs"
    model: str = "htdemucs"
    shifts: int = 1
    overlap: float = 0.25
    segment_length: float = 7.8

@dataclass
class TranscriptionConfig:
    model: str = "small"
    language: Optional[str] = None
    vad: bool = True
    condition_on_previous_text: bool = False
    beam_size: int = 5
    temperature: float = 0.0

@dataclass
class AudioConfig:
    normalize_peak: bool = True
    high_pass_filter: bool = False

@dataclass
class ExecutionConfig:
    force: bool = False
    mode: str = "full"  # 'full', 'separation-only', 'transcription-only'

@dataclass
class AppConfig:
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    logs_dir: Path = Path("logs")
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

def load_config(config_path: Optional[Path] = None, cli_overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
    """
    Loads AppConfig respecting hierarchy:
    CLI overrides > config.yaml > Defaults
    """
    yaml_data: Dict[str, Any] = {}
    
    if config_path is None:
        default_yaml = Path("config.yaml")
        if default_yaml.is_file():
            config_path = default_yaml

    if config_path and config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_data = loaded
        except Exception as e:
            print(f"[Aviso] No se pudo leer {config_path}: {e}. Se usarán valores por defecto.")

    # Defaults base
    app_cfg = AppConfig()

    # Apply YAML
    if "input_dir" in yaml_data:
        app_cfg.input_dir = Path(yaml_data["input_dir"])
    if "output_dir" in yaml_data:
        app_cfg.output_dir = Path(yaml_data["output_dir"])
    if "logs_dir" in yaml_data:
        app_cfg.logs_dir = Path(yaml_data["logs_dir"])

    hw = yaml_data.get("hardware", {})
    if "device" in hw:
        app_cfg.hardware.device = str(hw["device"]).lower()
    if "num_threads" in hw:
        app_cfg.hardware.num_threads = int(hw["num_threads"])

    sep = yaml_data.get("separation", {})
    if "backend" in sep:
        app_cfg.separation.backend = str(sep["backend"])
    if "model" in sep:
        app_cfg.separation.model = str(sep["model"])
    if "shifts" in sep:
        app_cfg.separation.shifts = int(sep["shifts"])
    if "overlap" in sep:
        app_cfg.separation.overlap = float(sep["overlap"])
    if "segment_length" in sep:
        app_cfg.separation.segment_length = float(sep["segment_length"])

    tra = yaml_data.get("transcription", {})
    if "model" in tra:
        app_cfg.transcription.model = str(tra["model"])
    if "language" in tra:
        app_cfg.transcription.language = tra["language"]
    if "vad" in tra:
        app_cfg.transcription.vad = bool(tra["vad"])
    if "condition_on_previous_text" in tra:
        app_cfg.transcription.condition_on_previous_text = bool(tra["condition_on_previous_text"])
    if "beam_size" in tra:
        app_cfg.transcription.beam_size = int(tra["beam_size"])
    if "temperature" in tra:
        app_cfg.transcription.temperature = float(tra["temperature"])

    aud = yaml_data.get("audio", {})
    if "normalize_peak" in aud:
        app_cfg.audio.normalize_peak = bool(aud["normalize_peak"])
    if "high_pass_filter" in aud:
        app_cfg.audio.high_pass_filter = bool(aud["high_pass_filter"])

    exe = yaml_data.get("execution", {})
    if "force" in exe:
        app_cfg.execution.force = bool(exe["force"])
    if "mode" in exe:
        app_cfg.execution.mode = str(exe["mode"])

    # Apply CLI Overrides (Highest priority)
    if cli_overrides:
        if cli_overrides.get("input_dir") is not None:
            app_cfg.input_dir = Path(cli_overrides["input_dir"])
        if cli_overrides.get("output_dir") is not None:
            app_cfg.output_dir = Path(cli_overrides["output_dir"])
        if cli_overrides.get("device") is not None:
            app_cfg.hardware.device = str(cli_overrides["device"]).lower()
        if cli_overrides.get("model") is not None:
            app_cfg.transcription.model = str(cli_overrides["model"])
        if cli_overrides.get("sep_model") is not None:
            app_cfg.separation.model = str(cli_overrides["sep_model"])
        if cli_overrides.get("language") is not None:
            app_cfg.transcription.language = cli_overrides["language"]
        if cli_overrides.get("force") is not None:
            app_cfg.execution.force = bool(cli_overrides["force"])
        if cli_overrides.get("separation_only"):
            app_cfg.execution.mode = "separation-only"
        elif cli_overrides.get("transcription_only"):
            app_cfg.execution.mode = "transcription-only"

    return app_cfg
