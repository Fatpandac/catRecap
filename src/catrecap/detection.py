"""在运动区域裁局部窗口，避免广角画面里的小宠物被整图缩放淹没。"""

import cv2


# ponytail: 每次最多四个固定比例窗口，避免在 Pi 上全图密集切片；拥挤场景仍可能漏掉较小运动。
MAX_REGIONS = 4
REGION_WIDTH = 0.2
REGION_HEIGHT = 0.3


def qualified_boxes(result, class_ids, confidence):
    return [
        box for box, cls, score in zip(
            result.boxes.xyxyn.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
        )
        if int(cls) in class_ids and score >= confidence
    ]


def motion_regions(mask):
    """返回掩码像素坐标下的裁剪窗口；掩码最大宽度由调用方限制为 640。"""
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted(
        (cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 12),
        key=lambda box: box[2] * box[3], reverse=True,
    )[:MAX_REGIONS]
    rw, rh = max(1, int(width * REGION_WIDTH)), max(1, int(height * REGION_HEIGHT))
    regions = []
    for x, y, w, h in boxes:
        x1 = max(0, min(width - rw, int(x + w / 2 - rw / 2)))
        y1 = max(0, min(height - rh, int(y + h / 2 - rh / 2)))
        regions.append((x1, y1, x1 + rw, y1 + rh))
    return regions


def detect_moving_pets(model, frame, mask, config, class_ids):
    """局部推理仍需满足物种与置信度门槛，输出坐标映射回整幅画面。"""
    height, width = frame.shape[:2]
    mh, mw = mask.shape
    boxes = []
    for left, top, right, bottom in motion_regions(mask):
        x1, x2 = round(left * width / mw), round(right * width / mw)
        y1, y2 = round(top * height / mh), round(bottom * height / mh)
        result = model.predict(
            frame[y1:y2, x1:x2], imgsz=config.detection_imgsz,
            conf=config.detection_confidence, classes=class_ids, verbose=False,
        )[0]
        for a, b, c, d in qualified_boxes(result, class_ids, config.detection_confidence):
            boxes.append((
                (x1 + a * (x2 - x1)) / width, (y1 + b * (y2 - y1)) / height,
                (x1 + c * (x2 - x1)) / width, (y1 + d * (y2 - y1)) / height,
            ))
    # 后续对像素掩码取并集，重复框不会重复累计运动面积，不额外维护 NMS/跟踪器。
    return boxes
