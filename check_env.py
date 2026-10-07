"""
Music Analyzer - Diagnostic and Environment Verification Script
Validates Python version, FFmpeg, PyTorch, CUDA, Demucs, Faster-Whisper and audio libraries.
"""

import sys
import shutil
import subprocess
from typing import Dict, Any, List

def check_python() -> Dict[str, Any]:
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    # Recommend Python 3.10 - 3.12 for optimal PyTorch / CTranslate2 stability
    ok = (v.major == 3 and 10 <= v.minor <= 12)
    warning = ""
    if v.major == 3 and v.minor > 12:
        warning = " (Advertencia: Python >= 3.13 puede tener soporte experimental en algunas librerías)"
    return {
        "name": "Python",
        "version": ver_str + warning,
        "ok": ok,
        "detail": sys.executable
    }

def check_ffmpeg() -> Dict[str, Any]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return {
            "name": "FFmpeg",
            "version": "No encontrado",
            "ok": False,
            "detail": "Instalar con: winget install Gyan.FFmpeg (Windows) o sudo apt install ffmpeg (Linux)"
        }
    try:
        res = subprocess.run([ffmpeg_path, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        first_line = res.stdout.splitlines()[0] if res.stdout else "Instalado"
        return {
            "name": "FFmpeg",
            "version": first_line.split(" Copyright")[0].strip(),
            "ok": True,
            "detail": ffmpeg_path
        }
    except Exception as e:
        return {
            "name": "FFmpeg",
            "version": "Error al ejecutar",
            "ok": False,
            "detail": str(e)
        }

def check_pytorch() -> Dict[str, Any]:
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_info = "Solo CPU"
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info = f"CUDA activo ({gpu_name}, {vram_gb:.1f} GB VRAM)"
        return {
            "name": "PyTorch",
            "version": f"{torch.__version__} [{gpu_info}]",
            "ok": True,
            "detail": f"CUDA Disponible: {cuda_ok}"
        }
    except ImportError:
        return {
            "name": "PyTorch",
            "version": "No instalado",
            "ok": False,
            "detail": "Instalar con: pip install torch torchaudio"
        }

def check_demucs() -> Dict[str, Any]:
    try:
        import demucs
        return {
            "name": "Demucs",
            "version": getattr(demucs, "__version__", "Instalado"),
            "ok": True,
            "detail": "Motor de separación de fuentes"
        }
    except ImportError:
        return {
            "name": "Demucs",
            "version": "No instalado",
            "ok": False,
            "detail": "Instalar con: pip install demucs"
        }

def check_faster_whisper() -> Dict[str, Any]:
    try:
        import faster_whisper
        return {
            "name": "Faster-Whisper",
            "version": getattr(faster_whisper, "__version__", "Instalado"),
            "ok": True,
            "detail": "Motor ASR local con CTranslate2"
        }
    except ImportError:
        return {
            "name": "Faster-Whisper",
            "version": "No instalado",
            "ok": False,
            "detail": "Instalar con: pip install faster-whisper"
        }

def check_audio_libs() -> List[Dict[str, Any]]:
    libs = [
        ("SoundFile", "soundfile"),
        ("Mutagen", "mutagen"),
        ("SciPy", "scipy"),
        ("PyYAML", "yaml"),
        ("Rich", "rich")
    ]
    results = []
    for display_name, mod_name in libs:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Instalado")
            results.append({"name": display_name, "version": str(ver), "ok": True, "detail": ""})
        except ImportError:
            results.append({
                "name": display_name,
                "version": "No instalado",
                "ok": False,
                "detail": f"pip install {mod_name}"
            })
    return results

def main():
    print("=" * 60)
    print(" MUSIC ANALYZER V1 - DIAGNÓSTICO DE ENTORNO")
    print("=" * 60)
    
    checks = [
        check_python(),
        check_ffmpeg(),
        check_pytorch(),
        check_demucs(),
        check_faster_whisper()
    ]
    checks.extend(check_audio_libs())

    all_critical_ok = True
    critical_names = {"Python", "FFmpeg", "PyTorch", "Demucs", "Faster-Whisper", "SoundFile", "Mutagen", "Rich", "PyYAML"}

    for item in checks:
        status_sym = "[OK]" if item["ok"] else "[X] "
        print(f" {status_sym} {item['name']:<16} : {item['version']}")
        if item["detail"] and not item["ok"]:
            print(f"      -> {item['detail']}")
        if item["name"] in critical_names and not item["ok"]:
            all_critical_ok = False

    print("-" * 60)
    if all_critical_ok:
        print("✓ Todos los componentes críticos están instalados y listos.")
        print("  El sistema puede procesar canciones en modo CPU (o CUDA si está disponible).")
    else:
        print("✗ Se encontraron dependencias faltantes o no compatibles.")
        print("  Ejecute: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
