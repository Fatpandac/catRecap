# CatRecap

从监控摄像头的视频流中识别宠物活动，截取 1080p 片段并推送到 Telegram。

## 处理流程

双流：720p 检测流负责"看"，1080p 高清流负责"录"，两边靠墙钟时间戳对齐。

```text
720p ch2 ──► 360p 前景分析 ──► 原图局部窗口 YOLO ──► 宠物框内运动确认
                                     │
                                 活动起止时间戳
                                     │
高清流 ch1 ──► ffmpeg -c copy 分段落盘 ──► 按时间戳 -c copy 剪片 ──► Telegram
```

高清录制走 `-c copy`，不重编码；检测流默认每 0.4 秒分析一次。注意 OpenCV 的 FFmpeg 后端
在 `grab()` 时仍会解码，采样主要减少颜色转换、拷贝和检测开销，并非只解码采样帧。

- 默认 `TRIGGER_MODE=pet_motion`：YOLO 必须确认配置的宠物类别，且运动必须发生在宠物检测框内；未知运动不触发，绝不自动退回通用运动。
- `TRIGGER_MODE=motion`：仅用于显式选择的通用运动监控，不要求宠物存在，因此不能用于保证只发宠物活动。
- `TRIGGER_MODE=yolo`：保留原先的猫狗检测 + 质心位移判定。
- 各模式共用 `src/catrecap/activity.py` 的事件起停、静止延时与冷却逻辑。
- 片段范围 = 活动开始前 `CLIP_PRE_SECONDS` 秒 到 判定停止的时刻（已含 `CLIP_POST_SECONDS` 的静止尾巴）。
  画面来自已落盘的录像，所以不需要在内存里缓冲视频。
- `CLIP_MAX_SECONDS` 截断长事件，`NOTIFICATION_COOLDOWN_SECONDS` 防止刷屏。
- 检测流断线每 5 秒重连；ffmpeg 录制进程挂掉会自动重启；推送失败只记日志，片段留在 `OUTPUT_DIR`，
  推送成功则删掉本地副本。
- 每 5 分钟一条心跳日志（读取帧率 / 源帧率 / 检测耗时 / 变化占比或宠物数 / 录像占用）。

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
cp .env.example .env      # 填 CAMERA_URL(推荐720p) / RECORD_URL(1080p) / TELEGRAM_*
uv run catrecap run
```

调试用法：

```sh
uv run catrecap run --dry-run                       # 照常录制和剪片，只是不推送
uv run catrecap run --debug --dry-run               # 开窗口看检测和判定，排查"猫动了却没发"
uv run catrecap run --source sample.mp4 --dry-run   # 回放本地视频调检测参数，片段直接从该文件剪
uv run python -m unittest discover -s tests -v
```

### 宠物运动模式（默认）：没有宠物证据就不触发

`.env` 可配置：

```dotenv
TRIGGER_MODE=pet_motion
PET_CLASSES=cat
DETECTION_CONFIDENCE=0.5
MOTION_MIN_RATIO=0.0008
MOTION_MAX_RATIO=0.5
MOTION_WARMUP_FRAMES=15
MOTION_IGNORE_PEOPLE=true
MOTION_PERSON_CONFIDENCE=0.4
MOTION_PERSON_MARGIN=0.5
MOTION_PIXEL_THRESHOLD=15
MOTION_CONFIRM_FRAMES=2
```

- 先把检测原图限制到最大宽 1280，再用最大宽 640 的小图做前景分析。从运动连通区域中心取最多 4 个局部窗口，回到原图裁切后运行 YOLO，最后把宠物框映射回整图。
  窗口占画面宽 20%、高 30%，候选按连通区域外接框面积排序；没有运动就不运行 YOLO，不做全图密集切片。
  本地 1080p 文件也走同样的缩放；当前机位推荐 `ch2` 的 720p，360p 原图在已标记的正样本上仍漏检。
- 宠物类别和置信度必须同时匹配。`PET_CLASSES=cat` 时，人、狗和低置信度猫框均不能满足宠物门控。
- **猫存在不等于猫在动**：只有宠物框内的运动参与连续确认；人在另一边走动不能让静止的猫满足条件。
  已确认的猫可以在人脚边，pet_motion 不再用扩大的“人体排除区”抹掉它。
- 没有宠物框时，运动掩码和连续确认计数清零，不用前几帧看到的猫给之后的人物活动放行。
- 这限制的是触发条件，不是画面裁剪；1080p 片段仍是完整画面，可能同时拍到人。
- 启动或检测流重连后，先学习 15 个采样帧的背景（默认约 6 秒），期间不触发新事件。
- MOG2 前景还必须满足相邻采样帧的灰度变化大于 `MOTION_PIXEL_THRESHOLD=15`，避免把已经停下的物体继续当作运动。
- 仅通用 motion 模式默认检测并排除前后两帧的人体框，向四周扩展最长边的 `50%`；`MOTION_PERSON_MARGIN` 控制扩展范围。
- 去噪并限制到宠物框后，变化像素占整图比例需在 `0.08%～50%` 内，且连续 `MOTION_CONFIRM_FRAMES=2` 个采样满足条件才触发。
  默认 0.4 秒采样下至少需要跨两个采样点，单次闪动会被过滤，短促的猫动作也可能漏检。
- `--debug` 黄框是达到置信度阈值的宠物区域，绿色是框内剩余的变化；没有黄框时显示 `NO PET: NO TRIGGER`。
  文字显示 `local pet detection`、`warmup`、`changed`、`confirm`、`ACTIVE/IDLE`、`cooldown`；蓝色人体排除框仅用于通用 motion。
  `ACTIVE` 表示事件尚未结束，并不表示已上传；结束后仍需等待剪辑与上传。
- 没触发先看预热和冷却，再看 `changed`：低于下限时可调小 `MOTION_MIN_RATIO`；误报多则调大。
- 宠物识别仍可能漏检或误分类；运动区域太多时猫可能排不到前 4 个候选，裁切也可能截断猫。宠物框内的人手或影子变化仍可能误报；这是证据门控，不是像素级物种分割或行为语义识别。
  若画面有猫却显示 `NO PET`，本模式选择不发送，需要用真实样本改善模型识别，不能再用一般运动代替猫的活动。
- `MOTION_IGNORE_PEOPLE`、`MOTION_PERSON_CONFIDENCE`、`MOTION_PERSON_MARGIN` 仅影响通用 motion。
  pet_motion 始终需要 YOLO 和宠物证据，不受这些人体参数影响；只有通用 motion 关闭人体过滤时才跳过 YOLO。
- 通用 motion 的人体过滤复用 `YOLO_MODEL` 和 `DETECTION_IMGSZ`，模型必须包含 `person` 类。
  pet_motion 不使用 `MOVE_THRESHOLD`；通用 motion 额外不使用 `PET_CLASSES` 或 `DETECTION_CONFIDENCE`。

按 `q` 退出窗口；`--dry-run` 仍会录制和保存片段，但不会发送。debug 的 `delivery SAVE ONLY (--dry-run)`
表示禁止发送，`delivery TELEGRAM ON (after event ends)` 表示事件结束后才会剪辑并尝试推送，**不代表已经发送成功**。
需要真正推送请去掉 `--dry-run`，使用 `uv run catrecap run --debug` 并重启进程。
通用 motion 不画猫框；pet_motion 画黄色宠物框和框内绿色运动区域，`ACTIVE` 表示事件进行中。本地文件自然播放结束时会收尾剪辑；手动退出时尚未结束的事件目前不会提交剪辑。
`--dry-run` 与发送失败留下的片段没有自动清理，调试后请检查 `OUTPUT_DIR`，避免长期积累。

### 已标记的真实视频回归

本地保留用户提供的两段私有视频（不入库、不上传），使用默认参数和 `PET_CLASSES=cat`：

```sh
CATRECAP_SAMPLE_TESTS=1 uv run python -m unittest discover -s tests -p test_labeled_samples.py
```

- `data/clips/pet-20260912-204107.mp4`：猫未活动，期望 0 个事件。
- `data/clips/pet-20260912-141219.mp4`：猫在人的脚边走动，期望 1 个事件，覆盖约 12 秒或 25 秒的猫活动；不同平台首次确认时刻可能不同。
- 测试真实解码、YOLO 和事件判定，仅拦截剪辑/上传；普通测试默认跳过这两个需要私有视频与本地权重的用例。
- 这两段用于修复回归，不是独立准确率评估集；不能据此保证其他光照、机位、遮挡下不漏检。树莓派实际推理耗时仍需实机测量。

### YOLO 模式的 `--debug`：为什么猫动了却没触发

仅在 `TRIGGER_MODE=yolo` 下适用。窗口画检测框，每次检测输出诊断日志，红字提示未满足的条件：

```text
pet 1 [cat 0.31]               ← 低阈值下的宠物框；none 不代表其他类别也没检出
conf>=0.50 passed 0            ← 达到 DETECTION_CONFIDENCE 的框数。0 = 置信度不够，调低阈值
move 0.0130 / 0.0200           ← 本次质心位移 / MOVE_THRESHOLD。偏小 = 猫动得不够"大"
state IDLE                     ← IDLE / ACTIVE（事件进行中）
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

### 宠物识别调参：采样对比工具

```sh
uv run python tools/probe.py collect --minutes 30   # 蹲守，把疑似有宠物的帧存到 data/samples/
uv run python tools/probe.py compare                # 对样本跑 模型 x imgsz 的组合
```

`collect` 只保存检测分数达到 0.08 的候选帧，会漏掉模型完全认不出的猫；没有样本不能证明画面没有猫。
这个工具存在采样偏差，不能用它单独估算识别准确率。需补充人工确认有猫、特别是模型漏检的原始画面。
它仍做整图 YOLO 比较，不包含 pet_motion 的运动局部窗口流程；实际触发请用上面的真实视频回归或 `run --source ... --dry-run` 验证。

`compare` 输出每个组合的「过阈值帧数 / 平均最高分 / 单帧耗时」，照着最划算的那行改 `.env` 的
`YOLO_MODEL` 和 `DETECTION_IMGSZ`。增大输入尺寸可能改善小目标检测，但无法恢复低清流已经丢失的细节；
是否有效必须在含猫样本上验证，性能必须在树莓派实测，不能用 Mac 数据代替。

pet_motion、yolo、启用人体过滤的 motion 或采样工具会加载模型，首次可能下载权重；只有关闭人体过滤的纯 motion 模式不加载 YOLO/torch。

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

Pi 5 上请实测心跳日志中的「读取 X 帧/秒（源 Y）」及检测耗时，Mac 数据不能代表 Pi 性能。
默认 pet_motion 每次采样运行 0～4 次局部窗口 YOLO（只检测宠物），心跳耗时包含本次所有窗口；如果跟不上实时，可调大 `DETECTION_INTERVAL`、
调小 `DETECTION_IMGSZ`，再考虑 NCNN：

```sh
uv run yolo export model=yolo11n.pt format=ncnn   # 生成 yolo11n_ncnn_model/
# 然后把 .env 里的 YOLO_MODEL 指向这个目录
```

ARM Linux 已在 `pyproject.toml` 中指定官方 CPU 版 torch/torchvision，`uv sync --locked` 不会为树莓派安装 CUDA 运行库。
服务文件里的 `User` 和 `/home/pi` 路径需替换为实际部署用户；也可直接用项目的 `.venv/bin/catrecap run` 作为 `ExecStart`，启动服务时无需重新同步依赖。

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
- YOLO 模式用**所有检测框的质心**判断位移，两只同时反向移动会互相抵消；motion 模式没有这个质心限制，但也不区分物种。
- 单条视频超过 Telegram 的 50MB 上限会推送失败，片段保留在本地。
- 暂无失败重试、无多摄像头、无 Web 界面，真需要再说。
