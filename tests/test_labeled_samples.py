"""真实机位回归：CATRECAP_SAMPLE_TESTS=1 uv run python -m unittest discover -s tests -p test_labeled_samples.py

视频是用户提供的私有样本，不入库，不上传。两段只是回归样本，不是独立准确率评估集。
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catrecap.config import Config
from catrecap import pipeline


@unittest.skipUnless(os.environ.get("CATRECAP_SAMPLE_TESTS") == "1", "需要显式启用和本地私有视频")
class LabeledSamplesTest(unittest.TestCase):
    def test_stationary_cat_does_not_trigger(self):
        self.check_sample("204107", expected_events=0)

    def test_walking_cat_triggers(self):
        self.check_sample("141219", expected_events=1)

    def check_sample(self, suffix, expected_events):
        source = Path(f"data/clips/pet-20260912-{suffix}.mp4")
        self.assertTrue(source.is_file(), f"缺少本地样本：{source}")
        self.assertTrue(Path("yolo11n.pt").is_file(), "请先准备本地 YOLO 权重")
        with tempfile.TemporaryDirectory() as temp:
            config = Config(
                str(source), "unused", "unused", output_dir=Path(temp), pet_classes=("cat",),
            )
            # 真实解码、真实 YOLO、真实事件判定；只拦截后续剪辑/上传，不发送消息。
            with patch.object(pipeline, "_clip_and_send") as clips:
                pipeline.run(config, dry_run=True)
            intervals = [(call.args[3], call.args[4]) for call in clips.call_args_list]
            self.assertEqual(len(intervals), expected_events, f"{suffix}: {intervals}")
            if expected_events:
                self.assertTrue(any(start < 12 < end for start, end in intervals), intervals)
            print(f"{suffix}: {len(intervals)} events, {intervals}")


if __name__ == "__main__":
    unittest.main()
