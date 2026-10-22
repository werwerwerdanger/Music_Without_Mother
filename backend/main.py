# -*- coding: utf-8 -*-
"""
StyleShift 后端

POST /api/convert   上传音频 + 风格，返回 task_id
GET  /api/task/{id} 查询进度和结果

启动:
    uvicorn main:app --host 127.0.0.1 --port 8000

依赖:
    fastapi uvicorn python-multipart librosa soundfile numpy scipy
"""

import os
import time
import uuid
import threading
import traceback
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
UPLOAD = BASE / "uploads"
OUTPUT = BASE / "outputs"
UPLOAD.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

app = FastAPI(title="StyleShift")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 任务表：{task_id: {status, progress, output, error}}
TASKS = {}

# ============================================================
# 风格预设
# ============================================================
STYLES = {
    # 古典：慢速、平衡 EQ、大混响、定音鼓
    "classical": dict(
        speed=0.60, sub=0.0, hi=0.0, reverb=0.55,
        drums="timpani", yell=False,
    ),
    # 通透：低频轰炸、砍高频、密集低频鼓、插怪叫
    "tongtou": dict(
        speed=1.00, sub=9.0, hi=-18.0, reverb=0.15,
        drums="sub_heavy", yell=True,
    ),
    # 8-bit：位深压缩、噪声鼓、亮
    "arcade": dict(
        speed=1.10, sub=-4.0, hi=8.0, reverb=0.02,
        drums="noise", bitcrush=True,
    ),
}


# ============================================================
# 音频处理
# ============================================================
def load_audio(path):
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def apply_eq(y, sr, sub_db, hi_db):
    """两段搁架式 EQ"""
    from scipy.signal import sosfilt, butter
    out = y.copy()
    if abs(sub_db) > 0.01:
        sos = butter(2, 150, btype="low", fs=sr, output="sos")
        low = sosfilt(sos, out)
        out = out + low * (10 ** (sub_db / 20) - 1)
    if abs(hi_db) > 0.01:
        sos = butter(2, 3000, btype="high", fs=sr, output="sos")
        high = sosfilt(sos, out)
        out = out + high * (10 ** (hi_db / 20) - 1)
    return out


def apply_reverb(y, sr, amount):
    """简易混响：多抽头延迟"""
    if amount <= 0.01:
        return y
    out = y.copy()
    taps = [(0.031, 0.55), (0.047, 0.42), (0.071, 0.32), (0.113, 0.22), (0.157, 0.14)]
    for delay, gain in taps:
        d = int(delay * sr)
        if d < len(y):
            out[d:] += y[:-d] * gain * amount
    return out


def detect_onsets(y, sr):
    """鼓点检测"""
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    frames = librosa.util.peak_pick(
        env, pre_max=3, post_max=3, pre_avg=5, post_avg=5,
        delta=0.35 * env.max() if env.max() > 0 else 0.1, wait=3,
    )
    return [f * 256 / sr for f in frames]


def synth_drum(y, sr, hits, kind):
    """按风格在鼓点位置生成新鼓"""
    out = y.copy()
    for t in hits:
        s0 = int(t * sr)
        if s0 >= len(out):
            continue

        if kind == "noise":
            # 8-bit：噪声脉冲 + 方波底鼓
            n = int(sr * 0.04)
            for i in range(min(n, len(out) - s0)):
                env = (1 - i / n) ** 3
                out[s0 + i] += (np.random.rand() * 2 - 1) * env * 0.45
            n2 = int(sr * 0.06)
            for i in range(min(n2, len(out) - s0)):
                env = (1 - i / n2) ** 2
                ph = (55 * i / sr) % 1
                out[s0 + i] += (1 if ph < 0.5 else -1) * env * 0.22

        elif kind == "sub_heavy":
            # 通透：低频正弦，每下都一样重
            f, n = 48, int(sr * 0.12)
            for i in range(min(n, len(out) - s0)):
                env = (1 - i / n) ** 2
                out[s0 + i] += np.sin(2 * np.pi * f * i / sr) * env * 0.65

        elif kind == "timpani":
            # 古典：定音鼓，只打强拍（每 4 下取 1）
            if round(t * 4) % 4 != 0:
                continue
            f0, n = 73, int(sr * 0.28)
            for i in range(min(n, len(out) - s0)):
                env = (1 - i / n) ** 2.2
                v = np.sin(2 * np.pi * f0 * i / sr) * 0.8
                v += np.sin(2 * np.pi * f0 * 1.5 * i / sr) * 0.3   # 五度泛音
                out[s0 + i] += v * env * 0.5

    return out


def insert_yells(y, sr, marks):
    """在标记处插入怪叫"""
    out = y.copy()
    for m in marks:
        if m.get("label") != "怪叫":
            continue
        s0 = int(float(m["time"]) * sr)
        if s0 >= len(out):
            continue
        n = int(sr * 0.35)
        f0 = 220 + np.random.rand() * 300
        sweep = (np.random.rand() - 0.5) * 2
        for i in range(min(n, len(out) - s0)):
            t = i / sr
            env = min(1, i / (sr * 0.02)) * (1 - i / n) ** 2
            f = f0 * (1 + sweep * t * 3)
            saw = ((f * t) % 1) * 2 - 1            # 锯齿波 = 人声感
            noise = (np.random.rand() * 2 - 1) * 0.4
            out[s0 + i] += (saw * 0.6 + noise) * env * 0.4
    return out


def bitcrush(y, bits=4, downsample=4):
    """位深压缩 + 降采样"""
    levels = 2 ** bits
    held = np.repeat(y[::downsample], downsample)[:len(y)]
    if len(held) < len(y):
        held = np.pad(held, (0, len(y) - len(held)), mode="edge")
    return np.round(held * levels) / levels


def process(src_path, style_id, marks, out_path, progress_cb):
    """完整处理管线"""
    import soundfile as sf

    cfg = STYLES[style_id]
    progress_cb(5, "加载音频")
    y, sr = load_audio(src_path)

    # 1. 变速
    if abs(cfg["speed"] - 1.0) > 0.01:
        progress_cb(15, f"变速 ×{cfg['speed']}")
        import librosa
        y = librosa.effects.time_stretch(y, rate=cfg["speed"])

    # 2. EQ
    progress_cb(30, "均衡")
    y = apply_eq(y, sr, cfg["sub"], cfg["hi"])

    # 3. 鼓点重写
    progress_cb(45, "鼓点重写")
    hits = detect_onsets(y, sr)
    y = synth_drum(y, sr, hits, cfg["drums"])

    # 4. 位深压缩（8-bit）
    if cfg.get("bitcrush"):
        progress_cb(60, "位深压缩")
        y = bitcrush(y, 4, 4)

    # 5. 怪叫（只在通透风格）
    if cfg.get("yell") and marks:
        progress_cb(70, "插入怪叫")
        y = insert_yells(y, sr, marks)

    # 6. 混响
    progress_cb(80, "混响")
    y = apply_reverb(y, sr, cfg["reverb"])

    # 7. 归一化
    progress_cb(90, "归一化")
    peak = np.abs(y).max()
    if peak > 0:
        y = y * (0.92 / peak)
    y = np.tanh(y * 1.1) * 0.95            # 软限幅

    sf.write(out_path, y.astype(np.float32), sr)
    progress_cb(100, "完成")
    return len(y) / sr


def run_task(task_id, src_path, style_id, marks, out_path):
    def cb(pct, msg):
        TASKS[task_id]["progress"] = pct
        TASKS[task_id]["message"] = msg

    try:
        TASKS[task_id]["status"] = "running"
        dur = process(src_path, style_id, marks, out_path, cb)
        TASKS[task_id].update(
            status="done",
            output=f"/api/audio/{task_id}",
            duration=round(dur, 1),
        )
    except Exception as e:
        TASKS[task_id].update(
            status="error",
            error=f"{e}",
            traceback=traceback.format_exc()[-800:],
        )


# ============================================================
# 路由
# ============================================================
@app.post("/api/convert")
async def convert(
    audio: UploadFile = File(...),
    style: str = Form(...),
    marks: str = Form("[]"),
):
    if style not in STYLES:
        raise HTTPException(400, f"未知风格: {style}")

    task_id = uuid.uuid4().hex[:12]
    ext = Path(audio.filename or "in.wav").suffix or ".wav"
    src = UPLOAD / f"{task_id}{ext}"
    with open(src, "wb") as f:
        f.write(await audio.read())

    import json
    try:
        mark_list = json.loads(marks)
    except Exception:
        mark_list = []

    out = OUTPUT / f"{task_id}.wav"
    TASKS[task_id] = {
        "status": "queued", "progress": 0, "message": "排队中",
        "style": style, "output": None, "created": time.time(),
    }

    threading.Thread(
        target=run_task, args=(task_id, src, style, mark_list, out), daemon=True
    ).start()

    return {"task_id": task_id}


@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    t = TASKS.get(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    return t


@app.get("/api/audio/{task_id}")
async def get_audio(task_id: str):
    t = TASKS.get(task_id)
    if not t or t["status"] != "done":
        raise HTTPException(404, "结果未就绪")
    path = OUTPUT / f"{task_id}.wav"
    if not path.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, media_type="audio/wav",
                        filename=f"styled_{task_id}.wav")


@app.get("/api/styles")
async def list_styles():
    return {
        k: {kk: vv for kk, vv in v.items() if kk != "yell"}
        for k, v in STYLES.items()
    }


@app.get("/api/health")
async def health():
    return {"ok": True, "tasks": len(TASKS)}


# 前端静态文件（如果存在）
FE = BASE.parent / "frontend"
if FE.exists():
    app.mount("/", StaticFiles(directory=str(FE), html=True), name="frontend")
