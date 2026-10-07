"""
Music Analyzer V1 - Main CLI Entry Point.
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import load_config
from src.hardware import detect_hardware
from src.logger import setup_logger
from src.pipeline import MusicPipeline

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music_analyzer",
        description="Music Analyzer V1: Separación local de fuentes (Vocals + Instrumental) y transcripción con timestamps."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Carpeta que contiene los archivos de audio a procesar (por defecto: 'input/')."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_dir",
        default=None,
        help="Directorio de destino para los resultados organizados (por defecto: 'output/')."
    )
    parser.add_argument(
        "-m", "--model",
        dest="model",
        choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
        default=None,
        help="Modelo de Faster-Whisper para transcripción (tiny, base, small, medium, large-v3, turbo)."
    )
    parser.add_argument(
        "--sep-model",
        dest="sep_model",
        default=None,
        help="Modelo de Demucs para separación (htdemucs, htdemucs_ft, htdemucs_6s)."
    )
    parser.add_argument(
        "-d", "--device",
        dest="device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Dispositivo de cómputo a utilizar (auto, cpu, cuda)."
    )
    parser.add_argument(
        "-l", "--language",
        dest="language",
        default=None,
        help="Forzar idioma para transcripción (código ISO, ej. 'es', 'en'). Si no se indica, se detecta automáticamente."
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=None,
        help="Forzar reprocesamiento de archivos aunque ya existan en la carpeta de salida."
    )
    parser.add_argument(
        "--separation-only",
        action="store_true",
        default=False,
        help="Ejecutar únicamente la etapa de separación (vocals.wav e instrumental.wav)."
    )
    parser.add_argument(
        "--transcription-only",
        action="store_true",
        default=False,
        help="Ejecutar únicamente la transcripción sobre los vocals.wav previamente generados."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Habilitar mensajes de depuración detallados en consola."
    )
    parser.add_argument(
        "--config",
        dest="config_file",
        default=None,
        help="Ruta a un archivo de configuración config.yaml personalizado."
    )
    return parser

def main():
    console = Console()
    parser = build_parser()
    args = parser.parse_args()

    cli_overrides = {
        "input_dir": args.input_path,
        "output_dir": args.output_dir,
        "model": args.model,
        "sep_model": args.sep_model,
        "device": args.device,
        "language": args.language,
        "force": args.force if args.force else None,
        "separation_only": args.separation_only,
        "transcription_only": args.transcription_only
    }
    # Clean None values
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    config_path = Path(args.config_file) if args.config_file else None
    app_config = load_config(config_path=config_path, cli_overrides=cli_overrides)

    setup_logger(verbose=args.verbose, log_dir=app_config.logs_dir)

    # Hardware Detection
    try:
        hw_info = detect_hardware(preferred_device=app_config.hardware.device)
    except Exception as e:
        console.print(f"[bold red]Error en detección de hardware:[/bold red] {e}")
        sys.exit(1)

    # Initial Info Banner
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[bold cyan]CPU:[/bold cyan]", f"{hw_info.cpu_name} ({hw_info.cpu_count} hilos)")
    table.add_row("[bold cyan]RAM:[/bold cyan]", f"{hw_info.ram_available_gb} GB disponibles / {hw_info.ram_total_gb} GB total")
    gpu_desc = f"{hw_info.gpu_name} ({hw_info.vram_total_gb} GB)" if hw_info.cuda_available else "No detectada (AMD/Integrada o sin CUDA)"
    table.add_row("[bold cyan]GPU CUDA:[/bold cyan]", gpu_desc)
    table.add_row("[bold cyan]Dispositivo Seleccionado:[/bold cyan]", f"[bold yellow]{hw_info.selected_device.upper()}[/bold yellow] (Whisper: {hw_info.whisper_compute_type})")
    table.add_row("[bold cyan]Modo de Ejecución:[/bold cyan]", f"[bold magenta]{app_config.execution.mode}[/bold magenta]")
    table.add_row("[bold cyan]Modelo Separador:[/bold cyan]", f"{app_config.separation.backend} ({app_config.separation.model})")
    table.add_row("[bold cyan]Modelo Transcriptor:[/bold cyan]", f"faster-whisper ({app_config.transcription.model})")
    table.add_row("[bold cyan]Carpeta Entrada:[/bold cyan]", str(app_config.input_dir))
    table.add_row("[bold cyan]Carpeta Salida:[/bold cyan]", str(app_config.output_dir))

    panel = Panel(
        table,
        title="[bold green]MUSIC ANALYZER V1[/bold green]",
        subtitle="[dim]Procesamiento Local Offline[/dim]",
        border_style="green"
    )
    console.print(panel)
    console.print("")

    # Run Pipeline
    pipeline = MusicPipeline(config=app_config, hardware=hw_info, console=console)
    summary = pipeline.run()

    # Final Summary Banner
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_row("[bold]Total canciones encontradas:[/bold]", str(summary.total_tracks))
    summary_table.add_row("[bold green]Completadas exitosamente:[/bold green]", str(summary.successful_tracks))
    
    if summary.failed_tracks:
        fail_str = f"[bold red]{len(summary.failed_tracks)}[/bold red]"
        summary_table.add_row("[bold red]Canciones con fallos:[/bold red]", fail_str)
        for name, err in summary.failed_tracks:
            summary_table.add_row(f"  [red]✗ {name}[/red]", f"[dim]{err}[/dim]")
    else:
        summary_table.add_row("[bold]Canciones con fallos:[/bold]", "[green]0[/green]")

    mins, secs = divmod(int(summary.elapsed_seconds), 60)
    summary_table.add_row("[bold]Tiempo total:[/bold]", f"{mins}m {secs}s")
    summary_table.add_row("[bold]Carpeta de resultados:[/bold]", str(app_config.output_dir.resolve()))
    summary_table.add_row("[bold]Archivo de log:[/bold]", str((app_config.logs_dir / "music_analyzer.log").resolve()))

    final_panel = Panel(
        summary_table,
        title="[bold green]RESUMEN DE PROCESAMIENTO[/bold green]",
        border_style="green" if not summary.failed_tracks else "yellow"
    )
    console.print(final_panel)

if __name__ == "__main__":
    main()
