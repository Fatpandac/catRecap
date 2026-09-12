"""命令行入口；视频处理流水线尚未接入。"""

import argparse
from importlib.metadata import version


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="catrecap",
        description="宠物活动视频摘要（初始化阶段，尚未接入视频处理与 Telegram）。",
    )
    parser.add_argument("--version", action="version", version=f"catrecap {version('catrecap')}")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
