"""
Orchestration pipeline for Music Analyzer V1.
Processes audio files sequentially, manages resume logic and isolates failures.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
from rich.console import Console

from src.config import AppConfig
from src.hardware import HardwareInfo, configure_torch_threads
from src.filesystem import (
    scan_audio_files,
    get_track_directories,
    has_valid_separation,
    has_valid_transcription
)
from src.audio import extract_metadata, save_wav, preprocess_vocals_for_transcription
from src.separators.demucs_separator import DemucsSeparator
from src.transcriber import WhisperTranscriber
from src.logger import get_logger

@dataclass
class PipelineSummary:
    total_tracks: int = 0
    successful_tracks: int = 0
    failed_tracks: List[Tuple[str, str]] = field(default_factory=list)
    skipped_stages: int = 0
    elapsed_seconds: float = 0.0

class MusicPipeline:
    """Sequential processing pipeline for albums and song collections."""

    def __init__(self, config: AppConfig, hardware: HardwareInfo, console: Optional[Console] = None):
        self.config = config
        self.hardware = hardware
        self.console = console or Console()
        self.logger = get_logger()

        # Adjust CPU threads
        configure_torch_threads(self.config.hardware.num_threads)

        # Lazy initialized models
        self.separator: Optional[DemucsSeparator] = None
        self.transcriber: Optional[WhisperTranscriber] = None

    def _get_separator(self) -> DemucsSeparator:
        if self.separator is None:
            self.separator = DemucsSeparator(
                model_name=self.config.separation.model,
                device=self.hardware.selected_device,
                shifts=self.config.separation.shifts,
                overlap=self.config.separation.overlap,
                segment_length=self.config.separation.segment_length
            )
        return self.separator

    def _get_transcriber(self) -> WhisperTranscriber:
        if self.transcriber is None:
            self.transcriber = WhisperTranscriber(
                model_size=self.config.transcription.model,
                device=self.hardware.selected_device,
                compute_type=self.hardware.whisper_compute_type,
                language=self.config.transcription.language,
                vad=self.config.transcription.vad,
                condition_on_previous_text=self.config.transcription.condition_on_previous_text,
                beam_size=self.config.transcription.beam_size,
                temperature=self.config.transcription.temperature
            )
        return self.transcriber

    def process_track(self, audio_path: Path) -> Tuple[bool, str]:
        """
        Processes a single track through separation and transcription.
        Returns (success: bool, status_message: str).
        """
        paths = get_track_directories(self.config.output_dir, audio_path.stem)
        force = self.config.execution.force
        mode = self.config.execution.mode

        # 1. Metadata extraction
        extract_metadata(audio_path, paths["metadata_json"])

        # 2. Separation stage
        if mode != "transcription-only":
            already_separated = has_valid_separation(paths)
            if already_separated and not force:
                self.logger.info(f"  [cyan]✓ Separación ya existe (omitida por reanudación)[/cyan]")
            else:
                self.logger.info(f"  [yellow]Separando fuentes (Demucs)...[/yellow]")
                sep = self._get_separator()
                sep_result = sep.separate(audio_path)
                save_wav(sep_result["vocals"], paths["vocals_wav"], sep_result["sample_rate"])
                save_wav(sep_result["instrumental"], paths["instrumental_wav"], sep_result["sample_rate"])
                self.logger.info(f"  [green]✓ Vocals e Instrumental guardados[/green]")

        # 3. Transcription stage
        if mode != "separation-only":
            if not paths["vocals_wav"].is_file():
                raise FileNotFoundError(f"No se encontró vocals.wav en {paths['vocals_wav']} para transcribir.")

            already_transcribed = has_valid_transcription(paths)
            if already_transcribed and not force:
                self.logger.info(f"  [cyan]✓ Transcripción ya existe (omitida por reanudación)[/cyan]")
            else:
                self.logger.info(f"  [yellow]Transcribiendo voces (Faster-Whisper)...[/yellow]")
                
                # Audio preprocessing (Peak norm / filter)
                preprocessed_path = paths["vocals_dir"] / "temp_asr_vocals.wav"
                asr_input = preprocess_vocals_for_transcription(
                    paths["vocals_wav"],
                    preprocessed_path,
                    normalize_peak=self.config.audio.normalize_peak,
                    high_pass_filter=self.config.audio.high_pass_filter
                )

                transcriber = self._get_transcriber()
                trans_result = transcriber.transcribe(asr_input)
                transcriber.save_transcription_outputs(trans_result, paths)

                # Clean temporary preprocessed file if created
                if preprocessed_path.exists() and preprocessed_path != paths["vocals_wav"]:
                    try:
                        preprocessed_path.unlink()
                    except Exception:
                        pass

                lang = trans_result.get("detected_language", "desconocido")
                prob = trans_result.get("language_probability", 0.0)
                self.logger.info(f"  [green]✓ Transcripción generada (.txt, .srt, .vtt) [idioma: {lang} ({prob*100:.0f}%)] [/green]")

        return True, "Completado"

    def run(self) -> PipelineSummary:
        """Executes processing across all discovered audio files."""
        files = scan_audio_files(self.config.input_dir)
        summary = PipelineSummary(total_tracks=len(files))
        start_time = time.time()

        if not files:
            self.console.print(f"[yellow]No se encontraron archivos compatibles en '{self.config.input_dir}'.[/yellow]")
            return summary

        self.console.print(f"[bold green]Encontradas {len(files)} canciones para procesar.[/bold green]\n")

        for idx, audio_file in enumerate(files, 1):
            track_name = audio_file.name
            self.console.print(f"[bold white][{idx}/{len(files)}] {track_name}[/bold white]")

            try:
                success, msg = self.process_track(audio_file)
                if success:
                    summary.successful_tracks += 1
            except Exception as e:
                self.logger.error(f"Error procesando {track_name}: {e}", exc_info=True)
                self.console.print(f"  [bold red]✗ Falló: {e}[/bold red]")
                summary.failed_tracks.append((track_name, str(e)))

            self.console.print("")

        summary.elapsed_seconds = round(time.time() - start_time, 2)
        return summary
