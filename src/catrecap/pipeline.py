"""低清流检测运动或运行 YOLO，高清流按事件时间戳无重编码剪片并推送。"""

import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

from catrecap import recorder
from catrecap.activity import ActivityTracker
from catrecap.config import Config
from catrecap.detection import detect_moving_pets, qualified_boxes as _qualified_boxes
from catrecap.motion import MotionDetector

log = logging.getLogger("catrecap")

TELEGRAM_VIDEO_LIMIT = 50 * 1024 * 1024  # Bot API sendVideo 上限
RECONNECT_DELAY = 5.0
HEARTBEAT_SECONDS = 300.0  # 心跳日志间隔：树莓派上判断是否跟得上实时
PRUNE_INTERVAL = 20.0  # 清理旧分段的节流间隔
SEGMENT_FLUSH_DELAY = 2.0  # 等 ffmpeg 把事件末尾的数据写进分段再剪
RECORDING_STALL_SECONDS = 30.0  # 不能只检查 PID：进程活着但停写也必须恢复
DEBUG_CONFIDENCE = 0.1  # debug 模式用更低的阈值跑，把被 conf 过滤掉的猫也显示出来
_gui_available = True  # headless 环境下置为 False，不要每帧都去试窗口


def run(
    config: Config, source: str | None = None, dry_run: bool = False, debug: bool = False
) -> None:
    source = source or config.camera_url
    is_stream = "://" in source
    if is_stream:
        # RTSP 默认走 UDP，丢包时画面会碎；树莓派上用 TCP 更稳。
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        # 起播时 FFmpeg 会刷一堆 non-existing PPS / decode_slice_header 噪音，只留 fatal。
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

    model, class_ids, person_ids, motion = None, [], [], None
    if config.trigger_mode != "motion" or config.motion_ignore_people:
        from ultralytics import YOLO

        model = YOLO(config.yolo_model)
        if config.trigger_mode != "motion":
            class_ids = _pet_class_ids(model.names, config.pet_classes)
        if config.trigger_mode == "motion" and config.motion_ignore_people:
            person_ids = _pet_class_ids(model.names, ("person",))
    log.info("触发模式：%s", config.trigger_mode)
    if config.trigger_mode == "motion":
        log.warning("motion 是通用运动模式，不能确认宠物；只关注宠物请使用 pet_motion")
    if dry_run:
        log.warning("dry-run：仅保存片段，不发送 Telegram；需要推送请移除 --dry-run 并重启")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tracker = ActivityTracker(
        move_threshold=config.move_threshold,
        post_seconds=config.clip_post_seconds,
        max_seconds=config.clip_max_seconds,
        cooldown=config.notification_cooldown_seconds,
    )
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip")
    recording = _Recording(config, source) if is_stream else None
    capture = None
    event_started_at = 0.0
    frame_index = 0
    last_detection_at = -1e9
    heartbeat_at, grabbed_frames, detections, detect_seconds = time.monotonic(), 0, 0, 0.0

    try:
        while True:
            if recording is not None:
                recording.tick()

            if capture is None:
                capture = cv2.VideoCapture(source)
                if not capture.isOpened():
                    capture.release()
                    capture = None
                    log.warning("打不开视频源，%.0fs 后重试", RECONNECT_DELAY)
                    time.sleep(RECONNECT_DELAY)
                    continue
                fps = _fps(capture)
                if config.trigger_mode in ("motion", "pet_motion"):
                    # 每次重连重新学习背景，避免旧画面触发伪事件。
                    motion = MotionDetector(
                        min_ratio=config.motion_min_ratio,
                        max_ratio=config.motion_max_ratio,
                        warmup_frames=config.motion_warmup_frames,
                        pixel_threshold=config.motion_pixel_threshold,
                        person_margin=config.motion_person_margin,
                        confirm_frames=config.motion_confirm_frames,
                    )
                log.info("已连接检测流，fps=%.1f", fps)

            # 只 grab 不 retrieve：非采样帧不做色彩转换和拷贝，也不让流缓冲堆积。
            if not capture.grab():
                capture.release()
                capture = None
                if not is_stream:
                    # 文件结束就收尾，不能等待一段已经不存在的静止尾巴。
                    if tracker.active:
                        worker.submit(
                            _clip_and_send, config, recording, source,
                            event_started_at, frame_index / fps, dry_run,
                        )
                    break  # 本地文件播放结束
                log.warning("视频流中断，%.0fs 后重连", RECONNECT_DELAY)
                time.sleep(RECONNECT_DELAY)
                continue

            frame_index += 1
            grabbed_frames += 1
            # 实时流用墙钟时间，丢帧不会让时间轴变慢；本地文件按帧号推算。
            timestamp = time.monotonic() if is_stream else frame_index / fps
            if timestamp - last_detection_at < config.detection_interval:
                continue
            last_detection_at = timestamp

            ok, frame = capture.retrieve()
            if not ok:
                continue

            detect_started = time.monotonic()
            result, centers, moving = None, [], None
            if config.trigger_mode == "pet_motion":
                # 识别原图最大宽 1280，前景图最大宽 640（16:9 时分别为 720p/360p）。
                # 本地 1080p 回放与摄像头 ch2 走相同预处理，避免离线和实机阈值不一致。
                height, width = frame.shape[:2]
                original = cv2.resize(frame, (1280, round(height * 1280 / width))) if width > 1280 else frame
                height, width = original.shape[:2]
                frame = cv2.resize(original, (640, round(height * 640 / width))) if width > 640 else original
                mask = motion.prepare(frame)
                pet_boxes = []
                if motion.frames > motion.warmup_frames:
                    pet_boxes = detect_moving_pets(model, original, mask, config, class_ids)
                # 已确认的猫可在人脚边，不能再被扩大的“人体排除区”清掉。
                moving = motion.evaluate(required_boxes=pet_boxes)
            elif motion is not None:
                ignored_boxes = []
                if model is not None:
                    result = _detect(
                        model, frame, config, person_ids, False,
                        confidence=config.motion_person_confidence,
                    )
                    ignored_boxes = _qualified_boxes(result, person_ids, config.motion_person_confidence)
                moving = motion.update(frame, ignored_boxes=ignored_boxes)
            else:
                result = _detect(model, frame, config, class_ids, debug)
                centers = _pet_centers(result, class_ids, config.detection_confidence)
            detect_seconds += time.monotonic() - detect_started
            detections += 1

            now = time.monotonic()
            if now - heartbeat_at >= HEARTBEAT_SECONDS:
                elapsed = now - heartbeat_at
                log.info(
                    "心跳：读取 %.1f 帧/秒（源 %.1f），检测 %.0fms/次，%s，录像占用 %s",
                    grabbed_frames / elapsed,
                    fps,
                    detect_seconds / max(detections, 1) * 1000,
                    f"变化占比 {motion.last_ratio:.3%}" if motion else f"宠物 {len(centers)} 只",
                    _disk_usage(config.segment_dir) if recording else "-",
                )
                heartbeat_at, grabbed_frames, detections, detect_seconds = now, 0, 0, 0.0

            event = tracker.update(timestamp, centers, moving=moving)

            if debug and not _show_debug_window(
                result, class_ids, tracker, config, timestamp,
                motion=motion, frame=frame, dry_run=dry_run,
            ):
                break  # 窗口里按了 q

            if event == "start":
                # 事件起点往前推 CLIP_PRE_SECONDS，画面已经在录像里，不用内存缓冲。
                event_started_at = _wall_clock(is_stream, timestamp) - config.clip_pre_seconds
                log.info("检测到活动（%s），标记起点 %s", config.trigger_mode, _fmt(event_started_at))
            elif event == "stop":
                event_ended_at = _wall_clock(is_stream, timestamp)
                worker.submit(
                    _clip_and_send, config, recording, source, event_started_at, event_ended_at, dry_run
                )
    finally:
        if capture is not None:
            capture.release()
        if debug and _gui_available:
            cv2.destroyAllWindows()
        worker.shutdown(wait=True)
        if recording is not None:
            recording.stop()


class _Recording:
    """管理 ffmpeg 分段录制子进程，并按时长/体积/磁盘余量清理旧分段。"""

    def __init__(self, config: Config, detection_source: str):
        self.config = config
        self.url = config.record_url or detection_source
        self.process = None
        self.pruned_at = 0.0
        self._start()

    def _start(self) -> None:
        self._last_write = None
        self._progress_at = time.monotonic()
        self.process = recorder.start_recording(
            self.url, self.config.segment_dir, self.config.segment_seconds
        )
        log.info("已启动高清录像，分段目录 %s", self.config.segment_dir)

    def tick(self) -> None:
        if self.process.poll() is not None:  # ffmpeg 挂了（断流、摄像头重启）
            log.warning("录像进程退出（code=%s），重启", self.process.returncode)
            time.sleep(RECONNECT_DELAY)
            self._start()
        now = time.monotonic()
        if now - self.pruned_at >= PRUNE_INTERVAL:
            self.pruned_at = now
            segments = recorder.list_segments(self.config.segment_dir)
            write = None
            if segments:
                path = segments[-1][0]
                try:
                    stat = path.stat()
                    if stat.st_size:
                        write = (path, stat.st_size, stat.st_mtime_ns)
                except FileNotFoundError:
                    pass
            if write is not None and write != self._last_write:
                self._last_write, self._progress_at = write, now
            elif now - self._progress_at >= RECORDING_STALL_SECONDS:
                log.warning("高清录像已 %.0f 秒没有写入，重启录制", now - self._progress_at)
                self.stop()
                self._start()
            recorder.prune_segments(
                self.config.segment_dir,
                keep_seconds=self.config.segment_keep_minutes * 60,
                max_bytes=int(self.config.segment_max_mb * 1024 * 1024),
                min_free_bytes=int(self.config.disk_min_free_mb * 1024 * 1024),
            )

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


def _clip_and_send(
    config: Config,
    recording: "_Recording | None",
    source: str,
    start: float,
    end: float,
    dry_run: bool,
) -> None:
    try:
        # 文件回放模式下 start 是片内偏移，不是墙钟时间，用当前时间命名。
        named_at = start if recording is not None else time.time()
        clip_path = config.output_dir / f"pet-{datetime.fromtimestamp(named_at):%Y%m%d-%H%M%S}.mp4"
        # end 是 tracker 判定停止的时刻，本身已含 CLIP_POST_SECONDS 的静止尾巴，不能再加一次，
        # 否则会去剪一段还没录到的“未来”，ffmpeg 只会把片段截短。
        if recording is not None:
            time.sleep(SEGMENT_FLUSH_DELAY)  # 事件末尾可能还在 ffmpeg 的写缓冲里
            ok = recorder.cut_clip(config.segment_dir, start, end, clip_path)
        else:
            ok = recorder.cut_file(Path(source), start, end, clip_path)
        if not ok:
            return
        log.info("已剪出 %s（目标 %.0f 秒）", clip_path.name, end - start)
        if dry_run:
            log.info("dry-run：跳过 Telegram 推送")
            return
        send_to_telegram(config, clip_path)
        clip_path.unlink(missing_ok=True)  # 推送成功就删本地副本，避免 output_dir 无限增长
        log.info("已推送 %s", clip_path.name)
    except Exception:
        # 推送失败保留本地文件，便于手动补发；失败不能拖垮取流循环。
        log.exception("剪辑或推送失败")


def _wall_clock(is_stream: bool, timestamp: float) -> float:
    # 流模式下 timestamp 是 monotonic，换算成剪辑需要的墙钟时间；文件模式本来就是片内偏移。
    return time.time() - (time.monotonic() - timestamp) if is_stream else timestamp


def _fmt(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S")


def _disk_usage(directory: Path) -> str:
    total = sum(path.stat().st_size for path, _ in recorder.list_segments(directory))
    return f"{total / 1024 / 1024:.0f}MB"


def _pet_class_ids(names: dict[int, str], pet_classes: tuple[str, ...]) -> list[int]:
    ids = [i for i, name in names.items() if name in pet_classes]
    if not ids:
        raise ValueError(f"模型不认识这些类别：{', '.join(pet_classes)}；可选：{sorted(names.values())}")
    return ids


def _detect(model, frame, config: Config, class_ids: list[int], debug: bool, *, confidence=None):
    # debug 时不限类别、阈值放到很低：否则分不清“模型什么都没看到”和“把猫认成了别的”。
    confidence = config.detection_confidence if confidence is None else confidence
    return model.predict(
        frame,
        imgsz=config.detection_imgsz,
        conf=min(confidence, DEBUG_CONFIDENCE) if debug else confidence,
        classes=None if debug else class_ids,
        verbose=False,
    )[0]


def _pet_centers(result, class_ids: list[int], confidence: float) -> list[tuple[float, float]]:
    """返回达到置信度阈值的宠物检测框中心点，归一化到 [0, 1]。"""
    return [
        (float(box[0]), float(box[1]))
        for box, conf, cls in zip(
            result.boxes.xywhn.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
        )
        if conf >= confidence and int(cls) in class_ids
    ]


def _show_debug_window(
    result, class_ids: list[int], tracker: ActivityTracker, config: Config, timestamp: float,
    *, motion: MotionDetector | None = None, frame=None, dry_run: bool = False,
) -> bool:
    """画一帧诊断画面；返回 False 表示用户要退出。"""
    if motion is not None:
        warming = motion.frames <= motion.warmup_frames
        within_range = motion.min_ratio <= motion.last_ratio <= motion.max_ratio
        lines = [(f"mode {config.trigger_mode}", False)]
        if config.trigger_mode == "pet_motion":
            lines += [
                (f"pet boxes >= {config.detection_confidence:.2f}: {len(motion.pet_boxes)} (yellow)",
                 not motion.pet_boxes),
                ("NO PET: NO TRIGGER" if not motion.pet_boxes else "pet region motion required", not motion.pet_boxes),
            ]
        else:
            lines.append(("NO PET CONFIRMATION", False))
        lines += [
            (("local pet detection; pet boxes take priority" if config.trigger_mode == "pet_motion" else
              f"person filter {'ON' if config.motion_ignore_people else 'OFF'}; masked regions {len(motion.ignored_boxes)} (blue)"), False),
            (f"warmup {min(motion.frames, motion.warmup_frames)}/{motion.warmup_frames}"
             f" {'WARMUP' if warming else 'READY'}", warming),
            (f"changed {motion.last_ratio:.3%} / min {motion.min_ratio:.3%}"
             f" max {motion.max_ratio:.1%}", not within_range),
            (f"confirm {min(motion.consecutive_frames, motion.confirm_frames)}/{motion.confirm_frames}",
             motion.consecutive_frames < motion.confirm_frames),
        ]
    else:
        boxes = list(zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()))
        detected = [f"{result.names[int(c)]} {v:.2f}" for c, v in boxes if int(c) in class_ids]
        others = sorted(
            (f"{result.names[int(c)]} {v:.2f}" for c, v in boxes if int(c) not in class_ids),
            key=lambda text: -float(text.split()[-1]),
        )
        passed = sum(1 for c, v in boxes if int(c) in class_ids and v >= config.detection_confidence)
        lines = [
            (f"pet {len(detected)} [{', '.join(detected[:3]) or 'none'}]", not detected),
            (f"other [{', '.join(others[:3]) or 'none'}]", False),
            (f"conf>={config.detection_confidence:.2f} passed {passed}", passed == 0),
            (f"move {tracker.last_move:.4f} / {config.move_threshold:.4f}",
             tracker.last_move <= config.move_threshold),
        ]
    cooldown_left = (
        max(0.0, config.notification_cooldown_seconds - (timestamp - tracker.stopped_at))
        if tracker.stopped_at is not None and not tracker.active
        else 0.0
    )
    # ACTIVE 表示事件进行中；高清录制本身始终运行，并非触发后才开始。
    lines += [
        (f"state {'ACTIVE' if tracker.active else 'IDLE'}", False),
        (f"cooldown {cooldown_left:.0f}s", cooldown_left > 0),
        ("delivery SAVE ONLY (--dry-run)" if dry_run else "delivery TELEGRAM ON (after event ends)",
         dry_run),
    ]
    log.debug(" | ".join(text for text, _ in lines))

    global _gui_available
    if not _gui_available:
        return True

    canvas = motion.overlay(frame) if motion else result.plot()
    for index, (text, blocking) in enumerate(lines):
        cv2.putText(
            canvas, text, (10, 24 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 255) if blocking else (0, 220, 0), 2, cv2.LINE_AA,
        )
    try:
        cv2.imshow("catRecap debug (q to quit)", canvas)
        return cv2.waitKey(1) & 0xFF != ord("q")
    except cv2.error:
        # 无 GUI 环境（树莓派 ssh、headless opencv）只留日志。
        _gui_available = False
        log.warning("当前环境打不开窗口，--debug 改为只输出诊断日志")
        return True


def _fps(capture) -> float:
    fps = capture.get(cv2.CAP_PROP_FPS)
    # 有些摄像头不报 fps 或报 0/nan，按 15 兜底，必要时按实际设备调。
    return fps if 0 < fps < 120 else 15.0


def send_to_telegram(config: Config, clip_path: Path) -> None:
    import requests

    size = clip_path.stat().st_size
    if size > TELEGRAM_VIDEO_LIMIT:
        raise ValueError(f"{clip_path.name} 有 {size / 1e6:.1f}MB，超过 Telegram 50MB 上限")
    if recorder.video_duration(clip_path, require_frame=True) < recorder.MIN_CLIP_SECONDS:
        raise ValueError(f"拒绝发送 {clip_path.name}：缺少至少 1 秒且首帧可解码的视频")
    with clip_path.open("rb") as video:
        response = requests.post(
            f"https://api.telegram.org/bot{config.telegram_bot_token}/sendVideo",
            data={
                "chat_id": config.telegram_chat_id,
                "caption": f"{'画面运动（未确认宠物）' if config.trigger_mode == 'motion' else '宠物活动'} {clip_path.stem}",
            },
            files={"video": (clip_path.name, video, "video/mp4")},
            timeout=120,
        )
    if not response.ok:
        # 不要打印 URL，里面带 bot token。
        raise RuntimeError(f"Telegram 返回 {response.status_code}: {response.text[:200]}")
