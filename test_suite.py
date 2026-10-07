"""
Test Suite for Music Analyzer V1.
Covers filesystem, metadata, audio DSP, formatters, and synthetic pipeline execution.
"""

import os
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

from src.filesystem import (
    sanitize_folder_name,
    scan_audio_files,
    get_track_directories,
    has_valid_separation,
    has_valid_transcription
)
from src.audio import extract_metadata, save_wav, preprocess_vocals_for_transcription
from src.transcriber import format_txt, format_srt, format_vtt
from src.config import load_config
from src.hardware import detect_hardware

class TestMusicAnalyzer(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("test_workspace")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        # Cleanup test workspace
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sanitize_folder_name(self):
        bad_name = 'Artist: Song / Track? *Awesome* <Version> | "Live"'
        clean = sanitize_folder_name(bad_name)
        # Check invalid characters were replaced
        for char in '<>:"/\\|?*':
            self.assertNotIn(char, clean)
        self.assertEqual(clean, "Artist_ Song _ Track_ _Awesome_ _Version_ _ _Live_")

    def test_transcription_formatters(self):
        segments = [
            {"start": 1.25, "end": 4.50, "text": "Primera línea de prueba"},
            {"start": 5.10, "end": 8.00, "text": "Segunda línea con rima"}
        ]
        
        txt = format_txt(segments)
        self.assertIn("[00:01.25] Primera línea de prueba", txt)
        self.assertIn("[00:05.10] Segunda línea con rima", txt)

        srt = format_srt(segments)
        self.assertIn("1\n00:00:01,250 --> 00:00:04,500\nPrimera línea de prueba", srt)
        self.assertIn("2\n00:00:05,100 --> 00:00:08,000\nSegunda línea con rima", srt)

        vtt = format_vtt(segments)
        self.assertTrue(vtt.startswith("WEBVTT\n"))
        self.assertIn("00:00:01.250 --> 00:00:04.500\nPrimera línea de prueba", vtt)

    def test_audio_metadata_and_save(self):
        # Create a synthetic 1-second stereo tone
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)
        stereo = np.column_stack([left, right])
        
        wav_path = self.test_dir / "test_synth.wav"
        save_wav(stereo, wav_path, sample_rate=sr)
        self.assertTrue(wav_path.exists())
        self.assertGreater(wav_path.stat().st_size, 1024)

        # Test metadata extraction
        meta_json = self.test_dir / "metadata.json"
        meta = extract_metadata(wav_path, meta_json)
        self.assertEqual(meta["title"], "test_synth")
        self.assertEqual(meta["channels"], 2)
        self.assertEqual(meta["sample_rate"], 44100)
        self.assertTrue(meta_json.exists())

    def test_audio_preprocessing(self):
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Low volume signal
        stereo = 0.1 * np.column_stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)])
        input_wav = self.test_dir / "low_vol.wav"
        save_wav(stereo, input_wav, sample_rate=sr)

        output_wav = self.test_dir / "norm.wav"
        res_path = preprocess_vocals_for_transcription(input_wav, output_wav, normalize_peak=True, high_pass_filter=True)
        self.assertTrue(res_path.exists())

        # Check that normalized peak is approximately -1 dBFS (~0.891)
        data, _ = sf.read(str(res_path))
        peak = np.max(np.abs(data))
        self.assertAlmostEqual(peak, 0.891, places=2)

    def test_resume_detection(self):
        out_root = self.test_dir / "output"
        dirs = get_track_directories(out_root, "track_01")
        
        self.assertFalse(has_valid_separation(dirs))
        self.assertFalse(has_valid_transcription(dirs))

        # Create dummy valid files
        dirs["vocals_wav"].write_bytes(b"RIFF" + b"\x00" * 2000)
        dirs["instrumental_wav"].write_bytes(b"RIFF" + b"\x00" * 2000)
        self.assertTrue(has_valid_separation(dirs))

        dirs["transcription_txt"].write_text("sample", encoding="utf-8")
        dirs["transcription_srt"].write_text("sample", encoding="utf-8")
        dirs["transcription_vtt"].write_text("sample", encoding="utf-8")
        self.assertTrue(has_valid_transcription(dirs))

if __name__ == "__main__":
    unittest.main()
