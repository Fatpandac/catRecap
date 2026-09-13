"""活动事件起停：接收运动标记，或根据检测框质心位移判定。"""

from dataclasses import dataclass, field
from math import dist


@dataclass
class ActivityTracker:
    """把逐帧的宠物检测结果，转换成 "start" / "stop" 活动事件。

    坐标使用归一化到 [0, 1] 的画面坐标，`move_threshold` 因此与分辨率无关。
    """

    move_threshold: float
    post_seconds: float
    max_seconds: float
    cooldown: float
    active: bool = False
    started_at: float | None = None
    # 下面几个是给 --debug 面板用的：最近一次位移、最后一次动的时刻、上次停止时刻。
    last_move: float = 0.0
    last_motion_at: float | None = None
    stopped_at: float | None = None
    _last_centroid: tuple[float, float] | None = field(default=None, repr=False)

    def update(
        self, timestamp: float, centers: list[tuple[float, float]], moving: bool | None = None
    ) -> str | None:
        """返回 "start"、"stop" 或 None；显式 moving 标记优先于质心判定。"""
        # ponytail: 用全部检测框的质心代表画面，多宠物同时反向移动会互相抵消；
        # 真要区分每只宠物，换成带 ID 的跟踪器（YOLO track）再按轨迹判断。
        centroid = _centroid(centers)
        self.last_move = (
            dist(centroid, self._last_centroid)
            if centroid is not None and self._last_centroid is not None
            else 0.0
        )
        if moving is None:
            moving = self.last_move > self.move_threshold
        self._last_centroid = centroid

        if self.active:
            if moving:
                self.last_motion_at = timestamp
            if (
                timestamp - self.last_motion_at > self.post_seconds
                or timestamp - self.started_at >= self.max_seconds
            ):
                self.active = False
                self.stopped_at = timestamp
                return "stop"
            return None

        if not moving:
            return None
        if self.stopped_at is not None and timestamp - self.stopped_at < self.cooldown:
            return None
        self.active = True
        self.started_at = timestamp
        self.last_motion_at = timestamp
        return "start"


def _centroid(centers: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not centers:
        return None
    return (
        sum(x for x, _ in centers) / len(centers),
        sum(y for _, y in centers) / len(centers),
    )
