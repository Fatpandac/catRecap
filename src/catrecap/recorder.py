"""用 ffmpeg 持续把高清流分段落盘，事件发生后按时间戳无重编码剪出片段。"""

import json
import logging
import math
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("catrecap")

SEGMENT_PREFIX = "seg-"
SEGMENT_SUFFIX = ".ts"  # mpegts 便于 concat，且正在写的段也能读
SEGMENT_PATTERN = f"{SEGMENT_PREFIX}%Y%m%d-%H%M%S{SEGMENT_SUFFIX}"
MIN_CLIP_SECONDS = 1.0


def start_recording(url: str, directory: Path, segment_seconds: float) -> subprocess.Popen:
    """启动 ffmpeg 分段录制子进程；`-c copy` 不解码不编码，几乎不吃 CPU。"""
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        # RTSP 无数据 15 秒即退出，让录制管理器重连，而不是活着但永远停写。
        "-rtsp_transport", "tcp", "-timeout", "15000000", "-i", url,
        "-c", "copy", "-an",
        "-f", "segment", "-segment_time", str(segment_seconds),
        # 必须传给分段内部的 MPEG-TS muxer，不能只设置外层 segment muxer。
        "-segment_format", "mpegts", "-segment_format_options", "flush_packets=1",
        "-reset_timestamps", "1", "-strftime", "1",
        str(directory / SEGMENT_PATTERN),
    ]
    return subprocess.Popen(command, stdin=subprocess.DEVNULL)


def parse_segment_time(path: Path) -> float | None:
    """从 seg-20260101-100000.ts 这样的文件名解析出段起始时间（本地时区 epoch）。"""
    if not path.name.startswith(SEGMENT_PREFIX) or path.suffix != SEGMENT_SUFFIX:
        return None
    try:
        stamp = datetime.strptime(path.stem[len(SEGMENT_PREFIX):], "%Y%m%d-%H%M%S")
    except ValueError:
        return None
    return stamp.timestamp()


def list_segments(directory: Path) -> list[tuple[Path, float]]:
    segments = []
    for path in directory.glob(f"{SEGMENT_PREFIX}*{SEGMENT_SUFFIX}"):
        started = parse_segment_time(path)
        if started is not None:
            segments.append((path, started))
    return sorted(segments, key=lambda item: item[1])


def select_segments(
    segments: list[tuple[Path, float]], start: float, end: float
) -> tuple[list[Path], float]:
    """按文件名挑候选段并返回相对偏移；不保证候选真正覆盖事件。

    最后一段时长未知，先选为候选；cut_clip 必须用实际媒体时长确认覆盖范围。
    """
    covering = []
    for index, (path, began) in enumerate(segments):
        # 相邻文件名仅作为候选边界；可能有断流缺口，实际覆盖由 cut_clip 再检查。
        finished = segments[index + 1][1] if index + 1 < len(segments) else float("inf")
        if began < end and finished > start:
            covering.append((path, began))
    if not covering:
        return [], 0.0
    # 录像可能还没覆盖到事件开头（比如刚启动），此时从最早那段的开头开始剪。
    return [path for path, _ in covering], max(0.0, start - covering[0][1])


def cut_clip(directory: Path, start: float, end: float, output: Path) -> bool:
    """从已落盘的分段里剪出 [start, end]，`-c copy` 不重编码。"""
    if end <= start:
        return False
    files, offset = select_segments(list_segments(directory), start, end)
    if not files:
        log.error("录像里没有覆盖 %s 的分段，无法剪辑", datetime.fromtimestamp(start))
        return False
    actual_start = max(start, parse_segment_time(files[0]))
    covered_until = actual_start
    for path in files:
        began = parse_segment_time(path)
        duration = video_duration(path)
        # 文件名仅精确到秒，允许 1 秒舍入误差；不能跨越断流造成的大段空白。
        if duration <= 0 or began > covered_until + 1:
            log.error("录像缺失或不连续，拒绝剪辑：%s", path.name)
            return False
        covered_until = max(covered_until, began + duration)
    if end > covered_until + 1:
        log.error("录像未覆盖事件尾部，缺少 %.1f 秒，拒绝剪辑", end - covered_until)
        return False
    listing = output.with_suffix(".txt")
    listing.write_text("".join(f"file '{f.resolve()}'\n" for f in files), encoding="utf-8")
    try:
        return _cut(["-f", "concat", "-safe", "0", "-i", str(listing)], offset, end - actual_start, output)
    finally:
        listing.unlink(missing_ok=True)


def cut_file(source: Path, start: float, end: float, output: Path) -> bool:
    """从本地视频文件里剪片段，供 `--source xxx.mp4` 回放调试用。"""
    return _cut(["-i", str(source)], max(start, 0.0), end - max(start, 0.0), output)


def video_duration(path: Path, *, require_frame: bool = False) -> float:
    """返回真实视频时长；缺视频流、解析失败或要求解码但首帧不可读时返回 0。

    只读一个视频包用于首帧校验，不把整段 1080p 再解码一遍。
    """
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    entries = "stream=width,height,duration:format=duration"
    if require_frame:
        command += ["-read_intervals", "%+#1"]
        entries += ":frame=width,height"
    command += ["-show_entries", entries, "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode:
            return 0.0
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        if stream.get("width", 0) <= 0 or stream.get("height", 0) <= 0:
            return 0.0
        raw_duration = stream.get("duration")
        if raw_duration in (None, "N/A"):
            raw_duration = info.get("format", {}).get("duration", 0)
        duration = float(raw_duration)
        if not math.isfinite(duration) or duration <= 0:
            return 0.0
        if require_frame and not any(
            frame.get("width", 0) > 0 and frame.get("height", 0) > 0
            for frame in info.get("frames", [])
        ):
            return 0.0
        return duration
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError, TypeError):
        return 0.0


def _cut(inputs: list[str], offset: float, duration: float, output: Path) -> bool:
    if not math.isfinite(duration) or duration <= 0:
        return False
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", *inputs,
        # ponytail: 输出侧 seek + copy，起点会对齐到关键帧，误差最多一个 GOP（常见 2 秒）。
        # 要精确到帧就得重编码，那正是这套方案想省掉的开销。
        "-ss", f"{offset:.3f}", "-t", f"{duration:.3f}",
        "-c", "copy", "-movflags", "+faststart", str(output),
    ]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if result.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        log.error("剪辑失败：%s", result.stderr.strip()[:300])
        output.unlink(missing_ok=True)
        return False
    if video_duration(output, require_frame=True) < MIN_CLIP_SECONDS:
        log.error("剪辑无效：%s 没有至少 1 秒且首帧可解码的视频，拒绝发送", output.name)
        output.unlink(missing_ok=True)
        return False
    return True


def prune_segments(
    directory: Path,
    keep_seconds: float,
    max_bytes: int,
    min_free_bytes: int = 0,
) -> None:
    """三道防线删旧分段：保留时长、占用上限、磁盘剩余空间。

    码率会随画面复杂度波动，只按时长算不住；任一道触发都从最旧的段开始删。
    永远至少留最新一段，否则刚录的画面会被自己删掉。
    """
    segments = list_segments(directory)
    deadline = time.time() - keep_seconds
    sizes = {path: path.stat().st_size for path, _ in segments if path.exists()}
    total = sum(sizes.values())
    free = shutil.disk_usage(directory).free

    for path, began in segments[:-1]:  # 最新一段永远保留
        if began >= deadline and total <= max_bytes and free >= min_free_bytes:
            break
        size = sizes.get(path, 0)
        path.unlink(missing_ok=True)
        total -= size
        free += size
        log.debug("清理旧分段 %s", path.name)
