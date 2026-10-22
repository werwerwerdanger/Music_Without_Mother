# -*- coding: utf-8 -*-
"""
程序化合成音色 —— 不依赖 SF2 音源

对比 GM 采样的优势:
  1. 完全可控（谐波、包络、颤音都能调）
  2. 不需要下载
  3. 参数化，能接进风格转换

用法:
    from synth import render_notes, Timbre
    render_notes(notes, sr, Timbre.STRINGS, "out.wav")
"""

import numpy as np
import wave
from dataclasses import dataclass


# ============================================================
# 基础波形
# ============================================================
def saw(freq, t, harmonics=12, rolloff=1.0):
    """锯齿波：所有谐波，幅度 1/k"""
    w = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        if freq * k > 16000:
            break
        w += np.sin(2 * np.pi * freq * k * t) / (k ** rolloff)
    return w / np.abs(w).max()


def square(freq, t, harmonics=10):
    """方波：奇次谐波"""
    w = np.zeros_like(t)
    for k in range(1, harmonics * 2, 2):
        if freq * k > 16000:
            break
        w += np.sin(2 * np.pi * freq * k * t) / k
    return w / np.abs(w).max()


def soft_saw(freq, t, harmonics=8):
    """柔和锯齿：谐波衰减快，接近弦乐"""
    w = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        if freq * k > 8000:
            break
        w += np.sin(2 * np.pi * freq * k * t + k * 0.3) / (k ** 1.8)
    return w / np.abs(w).max()


def sine_stack(freq, t, partials):
    """正弦叠加：partials = [(倍数, 幅度), ...]"""
    w = np.zeros_like(t)
    for mult, amp in partials:
        if freq * mult > 16000:
            continue
        w += np.sin(2 * np.pi * freq * mult * t) * amp
    return w / (np.abs(w).max() + 1e-9)


# ============================================================
# 包络
# ============================================================
def adsr(n, sr, a=0.02, d=0.1, s=0.7, r=0.3):
    """标准 ADSR 包络"""
    env = np.ones(n)
    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    na = min(na, n)
    if na > 0:
        env[:na] = np.linspace(0, 1, na)
    nd_end = min(na + nd, n)
    if nd_end > na:
        env[na:nd_end] = np.linspace(1, s, nd_end - na)
    if nd_end < n:
        env[nd_end:] = s
    # release 在末尾
    if nr > 0 and nr < n:
        env[-nr:] *= np.linspace(1, 0, nr)
    return env


def slow_attack(n, sr, attack=0.12, decay_tau=1.2):
    """慢起音 + 长衰减：弦乐感"""
    t = np.arange(n) / sr
    rise = np.minimum(t / attack, 1.0)
    fall = np.exp(-t / decay_tau)
    return rise * (0.35 + 0.65 * fall)


def pluck_env(n, sr, tau=0.08):
    """拨弦包络：极快衰减"""
    return np.exp(-np.arange(n) / sr / tau)


def perc_env(n, sr, tau=0.06, click=0.003):
    """打击包络：点击 + 指数衰减"""
    t = np.arange(n) / sr
    body = np.exp(-t / tau)
    c = int(click * sr)
    if c > 0:
        body[:c] += np.linspace(1, 0, c)
    return body


# ============================================================
# 音色定义
# ============================================================
@dataclass
class Timbre:
    name: str
    wave_fn: object        # (freq, t) -> array
    env_fn: object         # (n, sr) -> array
    vibrato: float = 0.0   # 颤音深度
    vibrato_rate: float = 5.5
    detune: float = 0.0    # 失谐（多层叠加）
    octave_layer: float = 0.0   # 八度叠加强度


def make_strings():
    """弦乐：柔和锯齿 + 慢起音 + 颤音 + 两层失谐"""
    return Timbre(
        name="strings",
        wave_fn=lambda f, t: soft_saw(f, t, 10),
        env_fn=lambda n, sr: slow_attack(n, sr, 0.10, 1.5),
        vibrato=0.004,
        detune=0.0025,
    )


def make_strings_long():
    """长弦乐：更慢的起音，更厚"""
    return Timbre(
        name="strings_long",
        wave_fn=lambda f, t: soft_saw(f, t, 12),
        env_fn=lambda n, sr: slow_attack(n, sr, 0.22, 2.2),
        vibrato=0.005,
        detune=0.004,
        octave_layer=0.25,
    )


def make_piano():
    """钢琴：谐波叠加 + 快衰减"""
    partials = [(1, 1.0), (2, 0.5), (3, 0.28), (4, 0.15), (5, 0.08), (6, 0.04)]
    return Timbre(
        name="piano",
        wave_fn=lambda f, t: sine_stack(f, t, partials),
        env_fn=lambda n, sr: pluck_env(n, sr, 0.35),
        vibrato=0.0,
    )


def make_pluck():
    """拨弦：快速衰减"""
    return Timbre(
        name="pluck",
        wave_fn=lambda f, t: soft_saw(f, t, 8),
        env_fn=lambda n, sr: pluck_env(n, sr, 0.12),
    )


def make_choir():
    """合唱：多个失谐的正弦叠加（模拟人声共振峰）"""
    partials = [(1, 1.0), (2, 0.6), (3, 0.35), (4, 0.2), (5, 0.12)]
    return Timbre(
        name="choir",
        wave_fn=lambda f, t: sine_stack(f, t, partials),
        env_fn=lambda n, sr: adsr(n, sr, 0.15, 0.2, 0.75, 0.4),
        vibrato=0.006,
        detune=0.004,
    )


def make_pad():
    """氛围垫：极慢起音"""
    return Timbre(
        name="pad",
        wave_fn=lambda f, t: soft_saw(f, t, 6),
        env_fn=lambda n, sr: slow_attack(n, sr, 0.5, 3.0),
        vibrato=0.003,
        detune=0.006,
        octave_layer=0.3,
    )


# ============================================================
# 渲染
# ============================================================
def hz(midi_note):
    return 440.0 * 2 ** ((midi_note - 69) / 12)


def render_note(freq, dur, sr, timbre, vel=90):
    """渲染单个音符

    颤音用相位调制实现：
        phase = 2π·f·t + depth·sin(2π·rate·t)
    而不是时间轴缩放 t*(1+depth*sin(...))，后者会让音高累积漂移。
    """
    n = max(64, int(dur * sr))
    t = np.arange(n) / sr

    # 相位调制（真正的颤音，不累积漂移）
    if timbre.vibrato > 0:
        phase = 2 * np.pi * freq * t + timbre.vibrato * np.sin(
            2 * np.pi * timbre.vibrato_rate * t
        )
        # 把相位转回等效时间，喂给波形函数
        t_mod = phase / (2 * np.pi * freq)
    else:
        t_mod = t

    sig = timbre.wave_fn(freq, t_mod)

    # 失谐层（模拟多人合奏）
    if timbre.detune > 0:
        sig2 = timbre.wave_fn(freq * (1 + timbre.detune), t_mod)
        sig = sig * 0.7 + sig2 * 0.5

    # 八度叠加
    if timbre.octave_layer > 0:
        sig += timbre.wave_fn(freq * 2, t_mod) * timbre.octave_layer

    env = timbre.env_fn(n, sr)
    return sig * env * (vel / 127.0)


def render_notes(notes, sr, timbre, path, bpm=120):
    """
    notes: [(beat_start, beat_dur, midi_note, velocity), ...]
    """
    if not notes:
        raise ValueError("没有音符")

    spb = 60.0 / bpm
    total_beats = max(b + d for b, d, p, v in notes)
    total = int(total_beats * spb * sr) + int(2 * sr)
    buf = np.zeros(total, dtype=np.float32)

    for beat, dur, pitch, vel in notes:
        s0 = int(beat * spb * sr)
        n = int(dur * spb * sr)
        if s0 >= total or n <= 0:
            continue
        sig = render_note(hz(pitch), n / sr, sr, timbre, vel)
        end = min(s0 + len(sig), total)
        buf[s0:end] += sig[:end - s0]

    # 归一化
    peak = np.abs(buf).max()
    if peak > 0:
        buf = buf * (0.9 / peak)
    buf = np.tanh(buf * 1.05) * 0.95

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((buf * 32767).astype(np.int16).tobytes())

    return len(buf) / sr


TIMBRES = {
    "strings": make_strings,
    "strings_long": make_strings_long,
    "piano": make_piano,
    "pluck": make_pluck,
    "choir": make_choir,
    "pad": make_pad,
}
