"""从 .env 和进程环境变量读取配置。"""

import os
from dataclasses import MISSING, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    camera_url: str
    telegram_bot_token: str
    telegram_chat_id: str
    # 默认值同时作为 .env 缺省时的回退，字段名大写即对应环境变量名。
    yolo_model: str = "yolo11n.pt"
    pet_classes: tuple[str, ...] = ("cat", "dog")
    detection_confidence: float = 0.5
    detection_interval: float = 0.4  # 每隔多少秒做一次推理，树莓派上别设太小
    detection_imgsz: int = 320
    clip_pre_seconds: float = 3.0
    clip_post_seconds: float = 5.0
    clip_max_seconds: float = 30.0
    move_threshold: float = 0.02  # 归一化画面坐标下的质心位移阈值
    notification_cooldown_seconds: float = 60.0
    output_dir: Path = Path("data/clips")


def load_config(env_path: Path = Path(".env"), environ=None) -> Config:
    values = _read_env_file(env_path)
    values.update(os.environ if environ is None else environ)

    kwargs = {}
    for field in fields(Config):
        raw = values.get(field.name.upper(), "").strip()
        if not raw:
            if field.default is not MISSING:
                continue  # 用 dataclass 默认值
            raise ValueError(f"缺少必填配置 {field.name.upper()}，请在 .env 中设置")
        kwargs[field.name] = _convert(field.name, field.type, raw)
    return Config(**kwargs)


def _read_env_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _convert(name: str, type_, raw: str):
    try:
        if type_ is float:
            return float(raw)
        if type_ is int:
            return int(raw)
        if type_ is Path:
            return Path(raw)
        if type_ == tuple[str, ...]:
            return tuple(item.strip() for item in raw.split(",") if item.strip())
        return raw
    except ValueError as exc:
        raise ValueError(f"配置 {name.upper()} 的值无效：{raw!r}（{exc}）") from exc
