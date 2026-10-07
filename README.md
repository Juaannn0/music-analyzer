# Music Analyzer V1
Herramienta de consola en Python para procesar álbumes y canciones almacenadas localmente. Separa fuentes de audio (voz e instrumental) mediante **Demucs v4** y genera transcripciones con marcas de tiempo (.txt, .srt, .vtt) mediante **Faster-Whisper**.

Diseñado para funcionar **100% de manera local, privada, gratuita y offline** en tu computador, sin costos por minuto ni llamadas a servicios en la nube.

---

## Características Principales

- **Separación de Fuentes Completa**: Produce tanto la pista vocal aislada (`vocals.wav`) como la pista instrumental sin voz (`instrumental.wav`).
- **Transcripción Local con Timestamps**: Genera letras sincronizadas con marcas de tiempo por línea (`[MM:SS.xx]`), subtítulos SubRip (`.srt`) y WebVTT (`.vtt`).
- **Filtrado Silero VAD**: Suprime alucinaciones de texto en secciones instrumentales y pausas vocales.
- **CPU como Ciudadano de Primera Clase**: Optimizado para funcionar fluidamente en procesadores modernos (como AMD Ryzen) utilizando cuantización `int8` y procesamiento secuencial (canción por canción).
- **Soporte Futuro para CUDA**: Detección automática de GPU NVIDIA con aceleración en `float16` sin alterar la arquitectura.
- **Reanudación Inteligente (Resume)**: Si un lote se interrumpe o falla una canción posterior, las pistas ya procesadas se omiten automáticamente, ahorrando tiempo y cómputo.
- **Extracción de Metadatos**: Lee etiquetas ID3 / Vorbis / MP4 y genera un `metadata.json` por cada canción.
- **Formatos Soportados**: MP3, WAV, FLAC, M4A, OGG, AAC y WMA.

---

## Estructura de Salida Generada

Por cada canción en la carpeta de entrada, se genera la siguiente estructura organizada:

```
output/
└── 01 - Nombre de Cancion/
    ├── metadata.json
    │
    ├── vocals/
    │   ├── vocals.wav
    │   ├── transcription.txt
    │   ├── transcription.srt
    │   └── transcription.vtt
    │
    └── instrumental/
        └── instrumental.wav
```

> **Nota sobre `instrumental.wav`**: Es un artefacto de primer orden. No solo permite escuchar la canción sin voces, sino que servirá como entrada para análisis musical avanzado en versiones futuras (detección de BPM, tonalidad armónica, percusión, etc.).

---

## Requisitos del Sistema

1. **Sistema Operativo**: Windows 10/11 (PowerShell / CMD) o Linux / WSL2.
2. **Procesador (CPU)**: CPU x86_64 multinúcleo (ej. AMD Ryzen 5 4600G o superior).
3. **Memoria RAM**: 16 GB recomendados.
4. **Almacenamiento**: ~3 GB de espacio para librerías y pesos de modelos.
5. **Python**: Python 3.10 a 3.12 (64-bit).
6. **FFmpeg**: Necesario para decodificar archivos de audio (MP3, M4A, etc.).

---

## Instalación Paso a Paso

### 1. Instalar FFmpeg (en Windows)

Si no tienes FFmpeg instalado, abre PowerShell y ejecuta:
```powershell
winget install Gyan.FFmpeg
```
*(O en Ubuntu/WSL: `sudo apt update && sudo apt install ffmpeg`)*

Cierra y vuelve a abrir tu terminal para que la variable `PATH` se actualice.

### 2. Clonar / Descargar el Proyecto y Crear el Entorno Virtual

En la raíz del proyecto:

**En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**En Linux / WSL2:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar Dependencias

**Si utilizas CPU (o gráficos integrados AMD Radeon Vega):**
```bash
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Si cuentas con una GPU NVIDIA con soporte CUDA:**
```bash
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Diagnóstico de Entorno

Ejecuta el script de diagnóstico para verificar que todos los componentes estén listos:
```bash
python check_env.py
```
Deberías ver todas las verificaciones con `[OK]`.

---

## Uso de la Herramienta (CLI)

Coloca tus archivos de música en la carpeta `input/` (o cualquier otra carpeta de tu elección).

### 1. Procesamiento Básico (Lote Completo)
```bash
python main.py input/
```

### 2. Seleccionar el Modelo de Whisper
Los modelos disponibles son `tiny`, `base`, `small` (recomendado para balance calidad/velocidad en CPU), `medium`, `large-v3` y `turbo`:
```bash
python main.py input/ --model small
```

### 3. Forzar Idioma (opcional)
Por defecto, el idioma se detecta automáticamente por canción. Para forzar un idioma:
```bash
python main.py input/ --language es
```

### 4. Selección de Dispositivo
```bash
# Automático (CUDA si existe, sino CPU)
python main.py input/ --device auto

# Forzar CPU
python main.py input/ --device cpu

# Forzar CUDA (arroja error si no hay GPU NVIDIA)
python main.py input/ --device cuda
```

### 5. Modos de Ejecución Parciales

**Solo Separación (Voz + Instrumental sin transcripción):**
```bash
python main.py input/ --separation-only
```

**Solo Transcripción (Usa los `vocals.wav` existentes):**
Útil para retranscribir con otro modelo de Whisper sin volver a esperar por la separación:
```bash
python main.py input/ --transcription-only --model medium
```

### 6. Forzar Reprocesamiento (`--force`)
Por defecto, si una canción ya tiene sus resultados en `output/`, se omite (*skip*). Para sobreescribir y volver a calcular:
```bash
python main.py input/ --force
```

### 7. Ver Ayuda Completa
```bash
python main.py --help
```

---

## Configuración Personalizada (`config.yaml`)

Puedes modificar los parámetros por defecto editando `config.yaml`:

```yaml
input_dir: "input"
output_dir: "output"

hardware:
  device: "auto"        # auto, cpu, cuda
  num_threads: 6        # Hilos recomendados para CPU Ryzen 5

separation:
  backend: "demucs"
  model: "htdemucs"
  shifts: 1
  overlap: 0.25

transcription:
  model: "small"        # tiny, base, small, medium, large-v3, turbo
  language: null        # null = autodetección
  vad: true             # Silero VAD activo
  condition_on_previous_text: false

audio:
  normalize_peak: true  # Normaliza picos a -1.0 dBFS
  high_pass_filter: false # Filtro pasa-altos a 80 Hz
```

> **Regla de precedencia**: Los argumentos pasados por consola (CLI) siempre tienen prioridad sobre los valores definidos en `config.yaml`.

---

## Pruebas del Sistema

Puedes ejecutar la suite de pruebas unitarias automatizadas:
```bash
python test_suite.py
```

---

## Limitaciones Conocidas

1. **Separación de Fuentes**: Los modelos neuronales no logran un aislamiento perfecto de estudio. Puede existir un sangrado leve (*bleed*) de percusión o sintetizadores en `vocals.wav` y ligeros ecos de coros en `instrumental.wav`.
2. **Transcripción Musical**: Whisper fue entrenado principalmente en lenguaje hablado. En canciones con rap ultra-rápido, jerga callejera (*slang*), Spanglish o voces saturadas de Auto-Tune, la transcripción puede contener errores que requieran edición en el archivo `.txt` o `.srt`.

---

## Licencia y Uso Responsable

Esta herramienta está pensada para propósitos de análisis personal, estudio musical y revisión de canciones que el usuario ya posea legalmente en su propio equipo. No incluye ni incluirá funciones de descarga ni evasión de protecciones digitales.
