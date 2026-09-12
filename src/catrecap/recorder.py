"""用 ffmpeg 持续把高清流分段落盘，事件发生后按时间戳无重编码剪出片段。"""

import logging
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("catrecap")

SEGMENT_PREFIX = "seg-"
SEGMENT_SUFFIX = ".ts"  # mpegts 便于 concat，且正在写的段也能读
SEGMENT_PATTERN = f"{SEGMENT_PREFIX}%Y%m%d-%H%M%S{SEGMENT_SUFFIX}"


def start_recording(url: str, directory: Path, segment_seconds: float) -> subprocess.Popen:
    """启动 ffmpeg 分段录制子进程；`-c copy` 不解码不编码，几乎不吃 CPU。"""
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-i", url,
        "-c", "copy", "-an",
        # 不缓写：事件刚结束就要去剪片，数据卡在 muxer 里会把尾巴剪丢。
        "-flush_packets", "1",
        "-f", "segment", "-segment_time", str(segment_seconds),
        "-segment_format", "mpegts", "-reset_timestamps", "1", "-strftime", "1",
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
    """挑出覆盖 [start, end] 的段，并返回剪辑起点相对第一段开头的偏移秒数。

    最后一段可能还在写，按"一直延伸到现在"处理。
    """
    covering = []
    for index, (path, began) in enumerate(segments):
        # 段的结束时间就是下一段的开始时间；最后一段仍在增长。
        finished = segments[index + 1][1] if index + 1 < len(segments) else float("inf")
        if began < end and finished > start:
            covering.append((path, began))
    if not covering:
        return [], 0.0
    # 录像可能还没覆盖到事件开头（比如刚启动），此时从最早那段的开头开始剪。
    return [path for path, _ in covering], max(0.0, start - covering[0][1])


def cut_clip(directory: Path, start: float, end: float, output: Path) -> bool:
    """从已落盘的分段里剪出 [start, end]，`-c copy` 不重编码。"""
    files, offset = select_segments(list_segments(directory), start, end)
    if not files:
        log.error("录像里没有覆盖 %s 的分段，无法剪辑", datetime.fromtimestamp(start))
        return False
    listing = output.with_suffix(".txt")
    listing.write_text("".join(f"file '{f.resolve()}'\n" for f in files), encoding="utf-8")
    try:
        return _cut(["-f", "concat", "-safe", "0", "-i", str(listing)], offset, end - start, output)
    finally:
        listing.unlink(missing_ok=True)


def cut_file(source: Path, start: float, end: float, output: Path) -> bool:
    """从本地视频文件里剪片段，供 `--source xxx.mp4` 回放调试用。"""
    return _cut(["-i", str(source)], max(start, 0.0), end - start, output)


def _cut(inputs: list[str], offset: float, duration: float, output: Path) -> bool:
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", *inputs,
        # ponytail: 输出侧 seek + copy，起点会对齐到关键帧，误差最多一个 GOP（常见 2 秒）。
        # 要精确到帧就得重编码，那正是这套方案想省掉的开销。
        "-ss", f"{offset:.3f}", "-t", f"{max(duration, 1.0):.3f}",
        "-c", "copy", "-movflags", "+faststart", str(output),
    ]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if result.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        log.error("剪辑失败：%s", result.stderr.strip()[:300])
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
