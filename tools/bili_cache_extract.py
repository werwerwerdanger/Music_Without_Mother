#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B 站客户端缓存 (m4s) 转音频

B 站 PC 客户端把视频缓存在:
    C:/Users/<user>/Videos/bilibili/<cid>/

每个目录下:
    <cid>-1-30016.m4s   视频流
    <cid>-1-30280.m4s   音频流
    videoInfo.json      元数据（标题、UP、时长）

坑: m4s 文件头前 9 个字节是 ASCII "000000000"，ffmpeg 直接读会报
    Invalid data found when processing input

用法:
    python bili_cache_extract.py <缓存目录> [输出目录]
    python bili_cache_extract.py --list          # 列出所有缓存
"""

import json
import subprocess
import sys
from pathlib import Path

CACHE_ROOT = Path.home() / "Videos" / "bilibili"
HEADER_BYTES = 9  # "000000000"


def list_cache():
    if not CACHE_ROOT.exists():
        print(f"缓存目录不存在: {CACHE_ROOT}")
        return []
    items = []
    for d in sorted(CACHE_ROOT.iterdir()):
        if not d.is_dir():
            continue
        info = d / "videoInfo.json"
        meta = {}
        if info.exists():
            try:
                meta = json.loads(info.read_text(encoding="utf-8"))
            except Exception:
                pass
        items.append((d, meta))
    return items


def show_list():
    items = list_cache()
    if not items:
        return
    print(f"{'目录':>14}  {'时长':>6}  {'标题'}")
    for d, meta in items:
        title = (meta.get("title") or "?")[:50]
        dur = meta.get("duration", "?")
        up = meta.get("uname", "?")
        print(f"{d.name:>14}  {dur:>6}  {title}")
        print(f"{'':>14}  {'':>6}  └ {up}")


def extract(cache_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    info = cache_dir / "videoInfo.json"
    meta = {}
    if info.exists():
        meta = json.loads(info.read_text(encoding="utf-8"))
    title = (meta.get("title") or cache_dir.name)[:40]
    # 文件名只留安全字符
    safe = "".join(c for c in title if c.isalnum() or c in " -_()（）")
    safe = safe.strip() or cache_dir.name

    # 优先 30280（音频），没有就用 30016
    candidates = sorted(cache_dir.glob("*-30280.m4s")) or sorted(cache_dir.glob("*-30016.m4s"))
    if not candidates:
        print(f"[跳过] {cache_dir.name}: 找不到 m4s")
        return False

    src = candidates[0]
    tmp = src.with_suffix(".tmp.m4a")
    with open(src, "rb") as f:
        f.seek(HEADER_BYTES)          # 跳过 "000000000"
        data = f.read()
    tmp.write_bytes(data)

    out = out_dir / f"{safe}.mp3"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
           "-vn", "-c:a", "libmp3lame", "-q:a", "2", str(out)]
    r = subprocess.run(cmd, capture_output=True)
    tmp.unlink(missing_ok=True)

    if r.returncode != 0:
        print(f"[失败] {cache_dir.name}: {r.stderr.decode()[:120]}")
        return False
    print(f"[完成] {out.name}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    if sys.argv[1] == "--list":
        show_list()
        return
    cache_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./out")

    if cache_dir.name.isdigit():
        cache_dir = CACHE_ROOT / cache_dir.name
    if not cache_dir.exists():
        print(f"目录不存在: {cache_dir}")
        return
    extract(cache_dir, out_dir)


if __name__ == "__main__":
    main()
