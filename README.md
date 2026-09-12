# CatRecap

从监控摄像头的视频流中识别宠物活动，截取 1080p 片段并推送到 Telegram。

## 处理流程

双流：低清流负责"看"，高清流负责"录"，两边靠墙钟时间戳对齐。

```text
低清流 ch3 ──► OpenCV 抽帧（每 DETECTION_INTERVAL 解一帧）──► YOLO 检测猫/狗
                                                              │
                                              质心位移 > 阈值 → 活动开始/结束时间戳
                                                              │
高清流 ch1 ──► ffmpeg -c copy 分段落盘（几乎不吃 CPU）──► 按时间戳 -c copy 剪片 ──► Telegram
```

为什么这么绕：`-c copy` 全程不解码不编码，1080p 录制近乎零 CPU；YOLO 只看 360p 且每 0.4 秒一帧。
树莓派上真正的开销只剩推理本身。

- 活动判定见 `src/catrecap/activity.py`：**位移即活动**，不区分吃饭、玩耍等具体行为。
- 片段范围 = 活动开始前 `CLIP_PRE_SECONDS` 秒 到 判定停止的时刻（已含 `CLIP_POST_SECONDS` 的静止尾巴）。
  画面来自已落盘的录像，所以不需要在内存里缓冲视频。
- `CLIP_MAX_SECONDS` 截断长事件，`NOTIFICATION_COOLDOWN_SECONDS` 防止刷屏。
- 检测流断线每 5 秒重连；ffmpeg 录制进程挂掉会自动重启；推送失败只记日志，片段留在 `OUTPUT_DIR`，
  推送成功则删掉本地副本。
- 每 5 分钟一条心跳日志（抽帧率 / 源帧率 / 推理耗时 / 录像占用），用来判断这台机器跟不跟得上实时。

## 磁盘保护

滚动录像是唯一会持续写盘的东西，`prune_segments` 每 20 秒跑一次，**三道防线同时生效**，
任一触发就从最旧的分段开始删（最新一段永远保留）：

- `SEGMENT_KEEP_MINUTES=10`：只留最近 10 分钟。
- `SEGMENT_MAX_MB=2048`：分段目录总占用上限，防止码率突然变高时算不准。
- `DISK_MIN_FREE_MB=1024`：磁盘剩余空间低于 1GB 就强制删，兜住别人把盘写满的情况。

1080p 约 1.1Mbps ≈ 500MB/小时，默认 10 分钟约 80MB。推送成功的片段会自动删除，失败的留在
`OUTPUT_DIR` 等人工处理——这部分没有自动清理，是故意的，免得丢掉唯一没发出去的证据。

树莓派用 SD 卡的话，可以把 `SEGMENT_DIR` 指到 tmpfs（如 `/dev/shm/catrecap`）减少写卡磨损，
8GB 内存分 1GB 足够放 10 分钟 1080p。

## 使用

需要 Python 3.12、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和 **ffmpeg**（录制和剪片都靠它）。

```sh
uv sync --locked
cp .env.example .env      # 填 CAMERA_URL(低清) / RECORD_URL(高清) / TELEGRAM_*
uv run catrecap run
```

调试用法：

```sh
uv run catrecap run --dry-run                       # 照常录制和剪片，只是不推送
uv run catrecap run --debug --dry-run               # 开窗口看检测和判定，排查"猫动了却没发"
uv run catrecap run --source sample.mp4 --dry-run   # 回放本地视频调检测参数，片段直接从该文件剪
uv run python -m unittest discover -s tests -v
```

### `--debug`：为什么猫动了却没触发

开一个实时窗口（ultralytics 画检测框）+ 每次检测一行日志，**红字就是挡住触发的那一项**：

```text
detect 1 [cat 0.31]            ← 低阈值下检到了什么。none = 模型根本没认出猫
conf>=0.50 passed 0            ← 达到 DETECTION_CONFIDENCE 的框数。0 = 置信度不够，调低阈值
move 0.0130 / 0.0200           ← 本次质心位移 / MOVE_THRESHOLD。偏小 = 猫动得不够"大"
state IDLE                     ← IDLE / RECORDING
cooldown 42s                   ← 冷却剩余。>0 时新活动不会触发，调小 NOTIFICATION_COOLDOWN_SECONDS
```

debug 模式下检测阈值临时降到 0.1，好让本来被过滤掉的框也显示出来；判定仍按 `.env` 的真实阈值走，
所以看到的行为和正常运行一致。窗口按 `q` 退出，刷新率就是 `DETECTION_INTERVAL`（默认 0.4 秒一帧，
看着卡是正常的，故意不改成每帧，否则复现不出真实判定节奏）。

常见结论：
- `detect none`：360p 里猫太小或夜视画面模型不认。试 `--source $RECORD_URL` 用 1080p 流跑检测，
  或把 `DETECTION_IMGSZ` 调到 640。
- `passed 0` 但 detect 有框：`DETECTION_CONFIDENCE` 降到 0.3 左右。
- `move` 一直小于阈值：猫在原地小幅动作，`MOVE_THRESHOLD` 调到 0.01 或更小。
- 一直 `cooldown`：上一段刚发完，属于预期，嫌少就调小冷却。

在树莓派上 ssh 跑 `--debug` 打不开窗口时，会自动降级成只输出上面那行日志。

首次运行会自动下载 `yolo11n.pt` 到工作目录。

## 树莓派 5 部署

```sh
sudo apt install -y ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo> /home/pi/catRecap && cd /home/pi/catRecap
uv sync --locked && cp .env.example .env && $EDITOR .env
uv run catrecap run --dry-run       # 先确认能连上两路流、能出片段

sudo cp deploy/catrecap.service /etc/systemd/system/
sudo systemctl enable --now catrecap
journalctl -u catrecap -f
```

Pi 5 纯 CPU 推理，`yolo11n.pt` 在 360p 上每次几十到几百毫秒。看心跳日志的「抽帧 X 帧/秒（源 Y）」：
X 明显低于 Y 就是跟不上实时，依次调大 `DETECTION_INTERVAL`、调小 `DETECTION_IMGSZ`，再考虑 NCNN：

```sh
uv run yolo export model=yolo11n.pt format=ncnn   # 生成 yolo11n_ncnn_model/
# 然后把 .env 里的 YOLO_MODEL 指向这个目录
```

torch 占磁盘约 1GB，8GB 的 Pi 5 装得下。

## 配置

所有配置项和默认值见 `.env.example` 与 `src/catrecap/config.py`；同名环境变量优先于 `.env`。
必填三项：`CAMERA_URL`、`TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`（私聊需先与 Bot 发起会话，
群组需先把 Bot 拉进去并给发送权限）。

`.env`、视频、模型权重和运行输出已在 `.gitignore` 中忽略。Telegram 推送会把家庭监控片段上传到
第三方服务，只在自己有权限的摄像头上使用。

## 已知限制

- 剪辑用 `-c copy`，起点对齐到关键帧，GOP 通常 2 秒 → 片段开头最多晚 2 秒、实际时长比目标短 1-2 秒。
  要精确到帧就得重编码，那正是这套方案想省掉的开销。
- 录制丢掉音轨（`-an`）：摄像头常用 G.711，mp4 容器 copy 不进去，留着会让剪辑直接失败。
- 多只宠物用**所有检测框的质心**判断位移，两只同时反向移动会互相抵消；要区分个体得换成带 ID 的跟踪。
- 单条视频超过 Telegram 的 50MB 上限会推送失败，片段保留在本地。
- 暂无失败重试、无多摄像头、无 Web 界面，真需要再说。
