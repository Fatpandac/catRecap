# CatRecap

从监控摄像头的视频流中识别宠物活动，截取活动片段并推送到 Telegram。

## 处理流程

```text
RTSP 视频流 → 每帧解码并进滚动缓冲 → 每隔 DETECTION_INTERVAL 跑一次 YOLO 检测猫/狗
  → 检测框质心位移超过阈值即判定为活动 → 截取活动前后的片段写成 mp4
  → Telegram Bot API sendVideo（后台线程，失败保留本地文件）
```

- 活动判定见 `src/catrecap/activity.py`：**位移即活动**，不区分吃饭、玩耍等具体行为。
- 片段包含事件前 `CLIP_PRE_SECONDS` 秒（滚动缓冲）和停止活动后 `CLIP_POST_SECONDS` 秒；
  `CLIP_MAX_SECONDS` 截断长事件，`NOTIFICATION_COOLDOWN_SECONDS` 防止刷屏。
- 视频源断线会每 5 秒重连；Telegram 推送失败只记日志，片段留在 `OUTPUT_DIR`。
- 每 5 分钟打一条心跳日志（处理帧率 / 源帧率 / 单次推理耗时）：处理帧率明显低于源帧率，
  说明这台机器跟不上实时，需要调大 `DETECTION_INTERVAL`、换 360p 通道或改用 NCNN 模型。

## 使用

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```sh
uv sync --locked
cp .env.example .env      # 填 CAMERA_URL / TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
uv run catrecap run
```

调试用法：

```sh
uv run catrecap run --source sample.mp4 --dry-run   # 回放本地视频，只存片段不推送
uv run catrecap run --source sample.mp4             # 回放并真的推送，验证 Telegram 链路
uv run python -m unittest discover -s tests -v
```

`--dry-run` 只跳过推送，仍会写片段到 `OUTPUT_DIR`。首次运行会自动下载 `yolo11n.pt` 到工作目录。

## 树莓派 5 部署

```sh
sudo apt install -y python3-pip
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo> /home/pi/catRecap && cd /home/pi/catRecap
uv sync --locked && cp .env.example .env && $EDITOR .env
uv run catrecap run --dry-run       # 先确认能连上摄像头、能出片段

sudo cp deploy/catrecap.service /etc/systemd/system/
sudo systemctl enable --now catrecap
journalctl -u catrecap -f
```

摄像头一般有多路码流（1080p/720p/360p），Pi 5 上优先选低分辨率那一路。

Pi 5 是纯 CPU 推理，torch 跑 `yolo11n.pt` 每帧约几十到上百毫秒。默认 `DETECTION_INTERVAL=0.4`、
`DETECTION_IMGSZ=320` 就是为此留的余量；如果 CPU 吃紧，先调大这两个值，再考虑导出 NCNN：

```sh
uv run yolo export model=yolo11n.pt format=ncnn   # 生成 yolo11n_ncnn_model/
# 然后把 .env 里的 YOLO_MODEL 指向这个目录
```

依赖里的 torch 占磁盘约 1GB，8GB 的 Pi 5 装得下；真嫌重可以改用 onnxruntime 自己做后处理，
但那要多写检测框解码和 NMS，现在没必要。

## 配置

所有配置项和默认值见 `.env.example` 与 `src/catrecap/config.py`；同名环境变量优先于 `.env`。
必填三项：`CAMERA_URL`、`TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`（私聊需先与 Bot 发起会话，
群组需先把 Bot 拉进去并给发送权限）。

`.env`、视频、模型权重和运行输出已在 `.gitignore` 中忽略。Telegram 推送会把家庭监控片段上传到
第三方服务，只在自己有权限的摄像头上使用。

## 已知限制

- 多只宠物用**所有检测框的质心**判断位移，两只同时反向移动会互相抵消；要区分个体得换成带 ID 的跟踪。
- 片段按视频源上报的 fps 写入；摄像头 fps 不准或抽帧时，片段时长会与真实时间有偏差。
- 优先用 H.264（`avc1`）编码，不可用时回落 `mp4v`，后者在部分客户端里不能内联预览。
- 单条视频超过 Telegram 的 50MB 上限会推送失败，片段保留在本地。
- 暂无失败重试、无多摄像头、无 Web 界面，真需要再说。
