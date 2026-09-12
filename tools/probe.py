"""采样真实画面并对比检测配置，用来决定 YOLO_MODEL / DETECTION_IMGSZ / DETECTION_CONFIDENCE。

    uv run python tools/probe.py collect --minutes 30   # 蹲守，把疑似有猫的帧存下来
    uv run python tools/probe.py compare                # 对样本跑多组配置，看谁检得准

样本存在 data/samples/（已被 .gitignore 忽略）。
"""

import argparse
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from catrecap.config import load_config

SAMPLE_DIR = Path("data/samples")
COLLECT_CONFIDENCE = 0.08  # 采样时阈值放到很低，宁可多存也别漏掉难例


def collect(config, url: str, minutes: float) -> None:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.yolo_model)
    capture = cv2.VideoCapture(url)
    deadline = time.time() + minutes * 60
    saved = 0
    last = 0.0
    while time.time() < deadline:
        if not capture.grab():
            time.sleep(1)
            continue
        if time.time() - last < 0.5:  # 每 0.5 秒看一帧就够，别存出一堆重复画面
            continue
        last = time.time()
        ok, frame = capture.retrieve()
        if not ok:
            continue
        result = model.predict(frame, imgsz=640, conf=COLLECT_CONFIDENCE, verbose=False)[0]
        hits = [
            (result.names[int(box.cls)], float(box.conf))
            for box in result.boxes
            if result.names[int(box.cls)] in config.pet_classes
        ]
        if not hits:
            continue
        name, conf = max(hits, key=lambda item: item[1])
        path = SAMPLE_DIR / f"{name}-{conf:.2f}-{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(str(path), frame)
        saved += 1
        print(f"存下 {path.name}")
    capture.release()
    print(f"共存 {saved} 张到 {SAMPLE_DIR}；没有样本说明这段时间模型一次都没认出宠物")


def compare(config, models: list[str], sizes: list[int]) -> None:
    samples = sorted(SAMPLE_DIR.glob("*.jpg"))
    if not samples:
        raise SystemExit(f"{SAMPLE_DIR} 里没有样本，先跑 collect")
    frames = [cv2.imread(str(path)) for path in samples]
    print(f"样本 {len(frames)} 张，类别 {config.pet_classes}\n")
    for model_name in models:
        model = YOLO(model_name)
        for imgsz in sizes:
            confs, elapsed = [], 0.0
            for frame in frames:
                started = time.time()
                result = model.predict(frame, imgsz=imgsz, conf=0.05, verbose=False)[0]
                elapsed += time.time() - started
                pet = [
                    float(box.conf)
                    for box in result.boxes
                    if result.names[int(box.cls)] in config.pet_classes
                ]
                confs.append(max(pet) if pet else 0.0)
            hit = sum(1 for c in confs if c >= config.detection_confidence)
            print(
                f"{model_name:<14} imgsz={imgsz:<4} "
                f"过阈值 {hit}/{len(frames)}  平均最高分 {sum(confs) / len(confs):.2f}  "
                f"{elapsed / len(frames) * 1000:.0f}ms/帧"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collector = sub.add_parser("collect")
    collector.add_argument("--minutes", type=float, default=30)
    collector.add_argument("--url", help="默认用 RECORD_URL（高清流，样本清晰度更高）")
    comparer = sub.add_parser("compare")
    comparer.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    comparer.add_argument("--sizes", nargs="+", type=int, default=[320, 640, 960])
    args = parser.parse_args()

    config = load_config()
    if args.command == "collect":
        collect(config, args.url or config.record_url or config.camera_url, args.minutes)
    else:
        compare(config, args.models, args.sizes)


if __name__ == "__main__":
    main()
