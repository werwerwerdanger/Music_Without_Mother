#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频基础分析：BPM、调性、响度、频谱质心

依赖: librosa, numpy
用法: python analyze.py <音频文件> [...]
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def analyze(path: Path):
    import librosa
    import numpy as np

    y, sr = librosa.load(path, sr=None, mono=True)
    dur = len(y) / sr
    print(f"\n{'=' * 60}\n{path.name}   {dur:.1f}s @ {sr}Hz")

    # BPM
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print(f"  BPM        {float(np.atleast_1d(tempo)[0]):.1f}")

    # 调性
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    prof = chroma.mean(axis=1)
    scores = []
    for i in range(12):
        for pt, mode in ((MAJOR, "major"), (MINOR, "minor")):
            r = np.corrcoef(np.roll(pt, i), prof)[0, 1]
            scores.append((r, NAMES[i], mode))
    scores.sort(reverse=True)
    top = scores[0]
    print(f"  调性       {top[1]} {top[2]}  (r={top[0]:.3f})")

    # 音级分布
    print("  音级       " + " ".join(f"{NAMES[i]}:{prof[i]:.2f}" for i in range(12)))

    # 频谱质心（亮度）
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    print(f"  频谱质心   {cent.mean():.0f} Hz")

    # 动态范围
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = 20 * np.log10(rms + 1e-9)
    print(f"  动态范围   {db.max() - db.min():.1f} dB  (peak {db.max():.1f} / floor {db.min():.1f})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            for f in sorted(p.glob("*.wav")) + sorted(p.glob("*.mp3")):
                analyze(f)
        elif p.exists():
            analyze(p)
        else:
            print(f"不存在: {p}")


if __name__ == "__main__":
    main()
