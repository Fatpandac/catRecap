import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from catrecap import pipeline
from catrecap.config import Config


def moving_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("Cannot create test video")
    try:
        for index in range(100):
            frame = np.full((240, 320, 3), 50, dtype=np.uint8)
            if 40 <= index < 60:
                x = 20 + (index - 40) * 8
                frame[100:130, x:x + 30] = 230
            writer.write(frame)
    finally:
        writer.release()


class MotionPipelineTest(unittest.TestCase):
    def test_pet_motion_requires_confident_pet_and_local_motion(self):
        names = {0: "person", 15: "cat", 16: "dog"}
        cases = (
            ("person_only", [0], [.9], [[0, 0, .7, 1]], 0),
            ("unknown_motion", [], [], [], 0),
            ("low_confidence_cat", [15], [.1], [[0, 0, .7, 1]], 0),
            ("dog_is_not_cat", [16], [.9], [[0, 0, .7, 1]], 0),
            ("person_moves_cat_elsewhere", [0, 15], [.9, .9], [[0, 0, .7, 1], [.8, 0, 1, 1]], 0),
            ("cat_region_moves", [15], [.9], [[0, 0, .7, 1]], 1),
            ("cat_moves_near_person", [0, 15], [.9, .9], [[0, 0, 1, 1], [0, 0, .7, 1]], 1),
        )
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "candidate.avi"
            moving_video(source)
            for name, classes, scores, boxes, expected_clips in cases:
                with self.subTest(name=name):
                    result = SimpleNamespace(boxes=SimpleNamespace(
                        cls=np.array(classes), conf=np.array(scores), xyxyn=np.array(boxes),
                    ))
                    model = SimpleNamespace(names=names, predict=lambda *a, **kw: [result])
                    config = Config(
                        str(source), "unused", "unused", output_dir=Path(temp) / "clips",
                        trigger_mode="pet_motion", pet_classes=("cat",),
                        detection_interval=.1, clip_post_seconds=1, motion_ignore_people=True,
                    )
                    with (
                        patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=lambda _: model)}),
                        patch.object(pipeline, "_clip_and_send") as clips,
                        self.assertLogs("catrecap", level="INFO"),
                    ):
                        pipeline.run(config, dry_run=True)
                    self.assertEqual(clips.call_count, expected_clips)

    def test_pipeline_masks_person_regions_before_triggering(self):
        # 人体推理由固定框代替；视频解码、运动检测、人体排除、事件状态机全部运行真实代码。
        result = SimpleNamespace(boxes=SimpleNamespace(
            xyxyn=np.array([[0, 0, 0.7, 1]]), cls=np.array([0]), conf=np.array([.9]),
        ))
        model = SimpleNamespace(names={0: "person"}, predict=lambda *args, **kwargs: [result])
        module = SimpleNamespace(YOLO=lambda path: model)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "person.avi"
            moving_video(source)
            config = Config(
                str(source), "unused", "unused", output_dir=Path(temp) / "clips",
                trigger_mode="motion", detection_interval=0.1, clip_post_seconds=1,
            )
            with (
                patch.dict(sys.modules, {"ultralytics": module}),
                patch.object(pipeline, "_clip_and_send") as clips,
                self.assertLogs("catrecap", level="INFO"),
            ):
                pipeline.run(config, dry_run=True)
            self.assertEqual(clips.call_count, 0, "人体区域内的运动不应触发剪辑")

    def test_active_event_is_clipped_at_file_end(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "unfinished.avi"
            moving_video(source)
            config = Config(
                str(source), "unused", "unused", output_dir=Path(temp) / "clips",
                trigger_mode="motion", motion_ignore_people=False,
                detection_interval=.1, clip_pre_seconds=1, clip_post_seconds=20,
            )
            with patch.object(pipeline, "_clip_and_send") as clips:
                pipeline.run(config, dry_run=True)
            self.assertEqual(clips.call_count, 1)
            self.assertLess(clips.call_args.args[3], 5)
            self.assertAlmostEqual(clips.call_args.args[4], 10)

    def test_debug_reports_sending_enabled_without_sending(self):
        config = Config("unused", "unused", "unused")
        tracker = pipeline.ActivityTracker(0.02, 5, 30, 60)
        motion = pipeline.MotionDetector()
        with (
            patch.object(pipeline, "_gui_available", False),
            self.assertLogs("catrecap", level="DEBUG") as logs,
        ):
            pipeline._show_debug_window(None, [], tracker, config, 0, motion=motion)
        self.assertTrue(any("delivery TELEGRAM ON (after event ends)" in line for line in logs.output))

    def test_motion_video_reaches_clipper_without_importing_yolo(self):
        # 真视频解码、背景建模、事件起停；仅替换剪辑/上传边界和 GUI 硬件调用。
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "motion.avi"
            moving_video(source)

            config = Config(
                str(source), "unused", "unused",
                output_dir=Path(temp) / "clips",
                detection_interval=0.1,
                trigger_mode="motion", motion_ignore_people=False,
                clip_pre_seconds=1,
                clip_post_seconds=1,
                clip_max_seconds=10,
            )
            with (
                patch.dict(sys.modules, {"ultralytics": None}),
                patch.object(pipeline, "_clip_and_send") as clips,
                patch.object(cv2, "imshow") as display,
                patch.object(cv2, "waitKey", return_value=-1),
                patch.object(cv2, "destroyAllWindows") as close,
                self.assertLogs("catrecap", level="DEBUG") as logs,
            ):
                try:
                    pipeline.run(config, dry_run=True, debug=True)
                except ModuleNotFoundError:
                    self.fail("motion mode must not import ultralytics")

            self.assertEqual(clips.call_count, 1)
            _, recording, actual_source, start, end, dry_run = clips.call_args.args
            self.assertIsNone(recording)
            self.assertEqual(actual_source, str(source))
            self.assertTrue(dry_run)
            # 两个连续采样确认后再起事件，前缓冲仍覆盖动作最初的画面。
            self.assertAlmostEqual(start, 3.4, delta=0.2)
            self.assertGreater(end, 6)
            self.assertLess(end, 9)
            self.assertTrue(display.called)
            close.assert_called_once()
            self.assertTrue(any("motion" in line and "WARMUP" in line for line in logs.output))
            self.assertTrue(any("ACTIVE" in line for line in logs.output))
            self.assertTrue(any("delivery SAVE ONLY (--dry-run)" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
