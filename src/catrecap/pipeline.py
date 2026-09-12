"""接视频流 → YOLO 检测宠物 → 截取活动片段 → 推送 Telegram。"""

import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

from catrecap.activity import ActivityTracker
from catrecap.config import Config

log = logging.getLogger("catrecap")

TELEGRAM_VIDEO_LIMIT = 50 * 1024 * 1024  # Bot API sendVideo 上限
RECONNECT_DELAY = 5.0


def run(config: Config, source: str | None = None, dry_run: bool = False) -> None:
    from ultralytics import YOLO  # 延迟导入：torch 启动慢，别拖累 --help

    source = source or config.camera_url
    is_stream = "://" in source
    if is_stream:
        # RTSP 默认走 UDP，丢包时画面会碎；树莓派上用 TCP 更稳。
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    model = YOLO(config.yolo_model)
    class_ids = _pet_class_ids(model.names, config.pet_classes)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tracker = ActivityTracker(
        move_threshold=config.move_threshold,
        post_seconds=config.clip_post_seconds,
        max_seconds=config.clip_max_seconds,
        cooldown=config.notification_cooldown_seconds,
    )
    sender = ThreadPoolExecutor(max_workers=1, thread_name_prefix="telegram")
    capture = writer = None
    clip_path = None
    frame_index = 0
    last_detection_at = -1e9

    try:
        while True:
            if capture is None:
                capture = cv2.VideoCapture(source)
                if not capture.isOpened():
                    capture.release()
                    capture = None
                    log.warning("打不开视频源，%.0fs 后重试", RECONNECT_DELAY)
                    time.sleep(RECONNECT_DELAY)
                    continue
                fps = _fps(capture)
                buffer = deque(maxlen=max(1, round(fps * config.clip_pre_seconds)))
                log.info("已连接视频源，fps=%.1f", fps)

            ok, frame = capture.read()
            if not ok:
                capture.release()
                capture = None
                if not is_stream:
                    break  # 本地文件播放结束
                log.warning("视频流中断，%.0fs 后重连", RECONNECT_DELAY)
                time.sleep(RECONNECT_DELAY)
                continue

            frame_index += 1
            # 实时流用墙钟时间，丢帧不会让时间轴变慢；本地文件按帧号推算。
            timestamp = time.monotonic() if is_stream else frame_index / fps
            buffer.append(frame)
            if writer is not None:
                writer.write(frame)

            if timestamp - last_detection_at < config.detection_interval:
                continue
            last_detection_at = timestamp

            event = tracker.update(timestamp, _pet_centers(model, frame, config, class_ids))
            if event == "start":
                clip_path = config.output_dir / f"pet-{datetime.now():%Y%m%d-%H%M%S}.mp4"
                writer = _open_writer(clip_path, fps, frame)
                if writer is None:
                    log.error("无法创建视频文件 %s，跳过本次事件", clip_path)
                    tracker.active = False
                    continue
                for buffered in buffer:  # 事件前若干秒，保留动作起点
                    writer.write(buffered)
                log.info("检测到宠物活动，开始录制 %s", clip_path.name)
            elif event == "stop" and writer is not None:
                writer.release()
                writer = None
                log.info("活动结束，已保存 %s", clip_path.name)
                if dry_run:
                    log.info("dry-run：跳过 Telegram 推送")
                else:
                    sender.submit(_send_safely, config, clip_path)
    finally:
        if writer is not None:
            writer.release()
            log.info("退出前保存 %s", clip_path.name)
        if capture is not None:
            capture.release()
        sender.shutdown(wait=True)


def _pet_class_ids(names: dict[int, str], pet_classes: tuple[str, ...]) -> list[int]:
    ids = [i for i, name in names.items() if name in pet_classes]
    if not ids:
        raise ValueError(f"模型不认识这些类别：{', '.join(pet_classes)}；可选：{sorted(names.values())}")
    return ids


def _pet_centers(model, frame, config: Config, class_ids: list[int]) -> list[tuple[float, float]]:
    """返回归一化到 [0, 1] 的宠物检测框中心点。"""
    result = model.predict(
        frame,
        imgsz=config.detection_imgsz,
        conf=config.detection_confidence,
        classes=class_ids,
        verbose=False,
    )[0]
    return [(float(x), float(y)) for x, y, _, _ in result.boxes.xywhn.tolist()]


def _fps(capture) -> float:
    fps = capture.get(cv2.CAP_PROP_FPS)
    # 有些摄像头不报 fps 或报 0/nan，按 15 兜底，必要时按实际设备调。
    return fps if 0 < fps < 120 else 15.0


def _open_writer(path: Path, fps: float, frame):
    height, width = frame.shape[:2]
    for codec in ("avc1", "mp4v"):  # H.264 优先，Telegram 里能直接预览
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def _send_safely(config: Config, clip_path: Path) -> None:
    try:
        send_to_telegram(config, clip_path)
        log.info("已推送 %s", clip_path.name)
    except Exception:
        # 推送失败保留本地文件，便于手动补发；失败不能拖垮取流循环。
        log.exception("推送失败，片段保留在 %s", clip_path)


def send_to_telegram(config: Config, clip_path: Path) -> None:
    import requests

    size = clip_path.stat().st_size
    if size > TELEGRAM_VIDEO_LIMIT:
        raise ValueError(f"{clip_path.name} 有 {size / 1e6:.1f}MB，超过 Telegram 50MB 上限")
    with clip_path.open("rb") as video:
        response = requests.post(
            f"https://api.telegram.org/bot{config.telegram_bot_token}/sendVideo",
            data={"chat_id": config.telegram_chat_id, "caption": f"宠物活动 {clip_path.stem}"},
            files={"video": (clip_path.name, video, "video/mp4")},
            timeout=120,
        )
    if not response.ok:
        # 不要打印 URL，里面带 bot token。
        raise RuntimeError(f"Telegram 返回 {response.status_code}: {response.text[:200]}")
