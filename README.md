# CatRecap

从监控摄像头的视频流中识别宠物活动，截取活动片段并推送到 Telegram。

## 当前状态

仓库初始化阶段：已有 Python 包、CLI 入口、配置模板和基础测试。
**尚未实现接流、YOLO 推理、活动判定、视频截取或 Telegram 推送。**
当前 CLI 仅提供帮助和版本信息，不会连接摄像头或发送消息，也不会读取 `.env`。

## 本地开发

使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```sh
uv sync --locked
cp .env.example .env
uv run catrecap --help
uv run catrecap --version
uv run python -m unittest discover -s tests -v
```

也可使用 `uv run python -m catrecap --help`。

初始化阶段不安装 PyTorch、YOLO 等较大依赖；接入推理时再根据运行设备选用 CPU、CUDA 或 MPS 环境，并锁定依赖。

## 计划中的处理流程

```text
摄像头 RTSP 视频流
  → 解码 / 滚动缓冲
  → YOLO 检测猫、狗
  → 连续帧活动判定 / 事件合并
  → 截取事件前后的视频，编码为 H.264 MP4
  → Telegram Bot API sendVideo
```

- 首版默认单摄像头、猫和狗，先跑通本地视频，再接实时 RTSP。
- 检测计划使用 Ultralytics YOLO，初始模型为 `yolo11n.pt`；接入前确认其 AGPL-3.0 / 商业许可是否符合用途。
- **目标检测不等于活动识别**：预训练 YOLO 可以定位猫狗，但不能直接判断吃饭、玩耍等行为。首版拟结合连续帧中的位置变化判断活动；具体行为分类需要额外模型或训练数据。
- 视频处理计划使用 OpenCV / FFmpeg；事件前缓冲保留动作起点，结束延时合并短暂漏检，片段时长上限与推送冷却避免刷屏。
- 推送直接使用 Telegram Bot API，不提前引入 Bot 框架、数据库或任务队列。实现时需限制文件大小、处理超时及限流，并保留发送失败的片段以便重试。
- 接流实现需处理断线重连、时间戳和缓冲区上限；本地片段保留策略需防止磁盘持续增长。

## 配置与隐私

`.env.example` 记录后续实现拟采用的配置项，**目前仅为模板**，尚未进行运行时读取或校验。

- `CAMERA_URL`：摄像头 RTSP 地址，可能包含账号密码。
- `TELEGRAM_BOT_TOKEN`：通过 BotFather 创建的 Bot token。
- `TELEGRAM_CHAT_ID`：接收消息的会话 ID；私聊需先与 Bot 发起会话，群组需先加入 Bot 并赋予发送权限。
- 模型、检测类别、置信度、片段前后时长及冷却参数见模板。

`.env`、视频、模型权重和运行输出已加入 `.gitignore`。不要将真实凭据写入源码、日志或测试；`.gitignore` 不是泄漏防护措施。Telegram 推送会将家庭监控片段上传至第三方，仅在明确授权的摄像头上使用。

## 下一步

1. 用本地样本验证 YOLO 检测及活动判定，明确是否仅关注移动还是需要具体行为分类。
2. 接入 RTSP，完成有界缓冲、事件片段生成和断线恢复。
3. 接入 Telegram，验证上传限制、失败保留和推送去重。

真实链路验证需要摄像头地址或视频样本、运行设备信息，以及 Telegram Bot 配置。
