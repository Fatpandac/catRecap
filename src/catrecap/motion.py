"""背景差分运动检测：可限制在已确认的宠物区域；过小的变化仍可能被过滤。"""

from dataclasses import dataclass, field

import cv2


@dataclass
class MotionDetector:
    """背景建模 + 前景像素占比判定画面里有没有东西在动。

    占比而不是像素数，因此和分辨率无关；上限用来忽略开灯、切夜视这种整幅突变。
    """

    min_ratio: float = 0.0008  # 前景占比下限，640×360 下约 184 个像素
    max_ratio: float = 0.5
    warmup_frames: int = 15  # 背景模型建起来之前一律不触发
    pixel_threshold: int = 15  # 相邻采样帧的灰度差（0～255），过滤压缩噪声
    person_margin: float = 0.5  # 人体框最长边的扩展比例，包含框外肢体和近处影子
    confirm_frames: int = 2
    consecutive_frames: int = 0
    last_ratio: float = 0.0
    frames: int = 0
    _mask = None
    _background: object = field(default=None, repr=False)
    _previous_gray: object = field(default=None, repr=False)
    _previous_boxes: tuple = field(default=(), repr=False)
    ignored_boxes: tuple = field(default=(), repr=False)
    pet_boxes: tuple = field(default=(), repr=False)

    def __post_init__(self):
        # ponytail: 只验证指定框内的像素变化，框内影子仍可能误报；行为语义需另加跟踪或模型。
        self._background = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False
        )

    def update(self, frame, ignored_boxes=(), *, required_boxes=None) -> bool:
        self.prepare(frame)
        return self.evaluate(ignored_boxes, required_boxes=required_boxes)

    def prepare(self, frame):
        """每个采样只更新一次背景和帧差，供局部检测使用，暂不触发事件。"""
        mask = self._background.apply(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is None:
            mask[:] = 0
        else:
            # 前景不等于运动：新物体停下后可能仍是前景，必须同时有相邻帧变化。
            _, changed = cv2.threshold(
                cv2.absdiff(gray, self._previous_gray), self.pixel_threshold, 255, cv2.THRESH_BINARY
            )
            mask = cv2.bitwise_and(mask, changed)
        self._previous_gray = gray
        self._mask = cv2.medianBlur(mask, 5)
        self.frames += 1
        return self._mask

    def evaluate(self, ignored_boxes=(), *, required_boxes=None) -> bool:
        """将本次检测结果限制到运动掩码，再做面积和连续帧确认。"""
        mask = self._mask
        # 同时排除前后两帧的人体位置，避免把人离开后的背景显露当成新运动。
        boxes = tuple(ignored_boxes)
        regions = []
        h, w = mask.shape
        for x1, y1, x2, y2 in boxes + self._previous_boxes:
            margin = max((x2 - x1) * w, (y2 - y1) * h) * self.person_margin
            x1, x2 = max(0, x1 - margin / w), min(1, x2 + margin / w)
            y1, y2 = max(0, y1 - margin / h), min(1, y2 + margin / h)
            mask[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 0
            regions.append((x1, y1, x2, y2))
        self.ignored_boxes = tuple(regions)
        self._previous_boxes = boxes
        self.pet_boxes = tuple(required_boxes) if required_boxes is not None else ()
        if required_boxes is not None:
            # None 表示通用运动；空列表表示没有确认宠物，必须清空运动掩码，不能退回通用运动。
            pet_mask = mask.copy()
            pet_mask[:] = 0
            for x1, y1, x2, y2 in self.pet_boxes:
                x1, x2 = int(max(0, x1) * w), int(min(1, x2) * w)
                y1, y2 = int(max(0, y1) * h), int(min(1, y2) * h)
                pet_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            mask = pet_mask
        self._mask = mask
        self.last_ratio = cv2.countNonZero(mask) / mask.size
        candidate = self.frames > self.warmup_frames and self.min_ratio <= self.last_ratio <= self.max_ratio
        self.consecutive_frames = self.consecutive_frames + 1 if candidate else 0
        return self.consecutive_frames >= self.confirm_frames

    def overlay(self, frame):
        """把前景区域涂成绿色，给 --debug 窗口看。"""
        if self._mask is None:
            return frame
        canvas = frame.copy()
        canvas[self._mask > 0] = (0, 255, 0)
        canvas = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)
        h, w = frame.shape[:2]
        for x1, y1, x2, y2 in self.pet_boxes:
            cv2.rectangle(canvas, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 255), 2)
        for x1, y1, x2, y2 in self.ignored_boxes:
            cv2.rectangle(canvas, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (255, 0, 0), 2)
        return canvas
