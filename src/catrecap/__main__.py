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
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from catrecap.config import load_config
    from catrecap.pipeline import run as run_pipeline

    try:
        config = load_config(args.env_file)
    except ValueError as exc:
        sys.exit(str(exc))
    try:
        run_pipeline(config, source=args.source, dry_run=args.dry_run)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
