"""
Hardware detection and device selection module for Music Analyzer V1.
Detects CPU cores, RAM, CUDA capability, GPU name/VRAM, and selects optimal compute types.
"""

import os
import platform
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareInfo:
    cpu_name: str
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    cuda_available: bool
    gpu_name: Optional[str]
    vram_total_gb: Optional[float]
    selected_device: str
    whisper_compute_type: str

def _get_ram_info() -> tuple[float, float]:
    """Retrieves total and available RAM in GB across Windows and Linux."""
    total_gb = 0.0
    available_gb = 0.0
    try:
        # Check Linux /proc/meminfo
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    info[parts[0].strip()] = parts[1].strip()
            total_kb = float(info.get("MemTotal", "0 kB").split()[0])
            avail_kb = float(info.get("MemAvailable", info.get("MemFree", "0 kB")).split()[0])
            total_gb = total_kb / (1024 * 1024)
            available_gb = avail_kb / (1024 * 1024)
        else:
            # Fallback for Windows
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_gb = stat.ullTotalPhys / (1024**3)
                available_gb = stat.ullAvailPhys / (1024**3)
    except Exception:
        # Reasonable fallback if os memory query fails
        total_gb = 16.0
        available_gb = 8.0
    return round(total_gb, 1), round(available_gb, 1)

def detect_hardware(preferred_device: str = "auto") -> HardwareInfo:
    """
    Detects hardware capabilities and selects execution device.
    Raises ValueError if CUDA is explicitly requested but not available.
    """
    pref = preferred_device.lower().strip()
    cpu_name = platform.processor() or platform.machine() or "CPU x86_64"
    cpu_count = os.cpu_count() or 6
    total_ram, avail_ram = _get_ram_info()

    cuda_available = False
    gpu_name = None
    vram_total_gb = None

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_total_gb = round(vram_bytes / (1024**3), 2)
    except ImportError:
        pass

    # Device selection logic
    if pref == "cuda":
        if not cuda_available:
            raise ValueError(
                "Se solicitó explícitamente el dispositivo 'cuda', pero no hay una GPU NVIDIA compatible con CUDA detectada en el entorno."
            )
        selected_device = "cuda"
        compute_type = "float16"
    elif pref == "cpu":
        selected_device = "cpu"
        compute_type = "int8"
    else:  # 'auto'
        if cuda_available:
            selected_device = "cuda"
            compute_type = "float16"
        else:
            selected_device = "cpu"
            compute_type = "int8"

    return HardwareInfo(
        cpu_name=cpu_name,
        cpu_count=cpu_count,
        ram_total_gb=total_ram,
        ram_available_gb=avail_ram,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        vram_total_gb=vram_total_gb,
        selected_device=selected_device,
        whisper_compute_type=compute_type
    )

def configure_torch_threads(num_threads: int = 6) -> None:
    """Configures PyTorch inter-op and intra-op thread counts for balanced CPU performance."""
    try:
        import torch
        threads = max(1, min(num_threads, os.cpu_count() or 6))
        torch.set_num_threads(threads)
    except ImportError:
        pass
