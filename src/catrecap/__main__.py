"""命令行入口。"""

import argparse
import logging
import sys
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="catrecap", description="监控视频里的宠物活动截取并推送 Telegram。")
    parser.add_argument("--version", action="version", version=f"catrecap {version('catrecap')}")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="开始监控视频源")
    run.add_argument("--source", help="覆盖 .env 里的 CAMERA_URL，可传本地视频文件用于回放调试")
    run.add_argument("--dry-run", action="store_true", help="只保存片段，不推送 Telegram")
    run.add_argument(
        "--debug",
        action="store_true",
        help="开一个窗口实时显示检测框和判定状态（红字 = 这一项挡住了触发）",
    )
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        return

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("catrecap").setLevel(logging.DEBUG if args.debug else logging.INFO)
    from catrecap.config import load_config
    from catrecap.pipeline import run as run_pipeline

    try:
        config = load_config(args.env_file)
    except ValueError as exc:
        sys.exit(str(exc))
    try:
        run_pipeline(config, source=args.source, dry_run=args.dry_run, debug=args.debug)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
