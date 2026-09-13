import tempfile
import unittest
from pathlib import Path

from catrecap.config import Config, load_config

MINIMAL = "CAMERA_URL=rtsp://cam/stream\nTELEGRAM_BOT_TOKEN=token\nTELEGRAM_CHAT_ID=42\n"


def write_env(text):
    path = Path(tempfile.mkdtemp()) / ".env"
    path.write_text(text, encoding="utf-8")
    return path


class LoadConfigTest(unittest.TestCase):
    def test_reads_values_and_applies_defaults(self):
        config = load_config(write_env(MINIMAL), environ={})
        self.assertEqual(config.camera_url, "rtsp://cam/stream")
        self.assertEqual(config.telegram_chat_id, "42")
        self.assertEqual(config.pet_classes, ("cat", "dog"))
        self.assertEqual(config.detection_confidence, Config.detection_confidence)
        self.assertEqual(config.clip_pre_seconds, Config.clip_pre_seconds)

    def test_parses_comments_quotes_and_typed_values(self):
        config = load_config(
            write_env(
                MINIMAL
                + "# 注释行\n"
                + '\nPET_CLASSES = "cat, bird "\n'
                + "DETECTION_CONFIDENCE=0.35\n"
                + "CLIP_MAX_SECONDS=20\n"
            ),
            environ={},
        )
        self.assertEqual(config.pet_classes, ("cat", "bird"))
        self.assertEqual(config.detection_confidence, 0.35)
        self.assertEqual(config.clip_max_seconds, 20.0)

    def test_real_environment_overrides_env_file(self):
        config = load_config(write_env(MINIMAL), environ={"TELEGRAM_CHAT_ID": "99"})
        self.assertEqual(config.telegram_chat_id, "99")

    def test_missing_env_file_still_reads_environment(self):
        config = load_config(
            Path(tempfile.mkdtemp()) / "absent.env",
            environ={
                "CAMERA_URL": "rtsp://cam/stream",
                "TELEGRAM_BOT_TOKEN": "token",
                "TELEGRAM_CHAT_ID": "42",
            },
        )
        self.assertEqual(config.camera_url, "rtsp://cam/stream")

    def test_missing_required_value_names_the_key(self):
        for text, missing in (
            (MINIMAL.replace("CAMERA_URL=rtsp://cam/stream\n", ""), "CAMERA_URL"),
            (MINIMAL.replace("TELEGRAM_BOT_TOKEN=token\n", ""), "TELEGRAM_BOT_TOKEN"),
            (MINIMAL.replace("TELEGRAM_CHAT_ID=42", "TELEGRAM_CHAT_ID="), "TELEGRAM_CHAT_ID"),
        ):
            with self.subTest(missing=missing):
                with self.assertRaises(ValueError) as ctx:
                    load_config(write_env(text), environ={})
                self.assertIn(missing, str(ctx.exception))

    def test_pet_motion_is_default_and_yolo_remains_selectable(self):
        self.assertEqual(load_config(write_env(MINIMAL), environ={}).trigger_mode, "pet_motion")
        config = load_config(
            write_env(MINIMAL),
            environ={"TRIGGER_MODE": "yolo", "MOTION_MIN_RATIO": "0.002"},
        )
        self.assertEqual(config.trigger_mode, "yolo")
        self.assertEqual(config.motion_min_ratio, 0.002)

    def test_person_filter_defaults_on_and_can_be_disabled(self):
        config = load_config(write_env(MINIMAL), environ={})
        self.assertTrue(config.motion_ignore_people)
        config = load_config(write_env(MINIMAL), environ={"MOTION_IGNORE_PEOPLE": "false"})
        self.assertIs(config.motion_ignore_people, False)
        with self.assertRaisesRegex(ValueError, "MOTION_IGNORE_PEOPLE"):
            load_config(write_env(MINIMAL), environ={"MOTION_IGNORE_PEOPLE": "typo"})

    def test_rejects_invalid_motion_settings(self):
        for values, key in (
            ({"TRIGGER_MODE": "typo"}, "TRIGGER_MODE"),
            ({"MOTION_MIN_RATIO": "0"}, "MOTION_MIN_RATIO"),
            ({"MOTION_MIN_RATIO": "nan"}, "MOTION_MIN_RATIO"),
            ({"MOTION_MIN_RATIO": "0.6", "MOTION_MAX_RATIO": "0.5"}, "MOTION_MIN_RATIO"),
            ({"MOTION_MAX_RATIO": "1.1"}, "MOTION_MIN_RATIO"),
            ({"MOTION_WARMUP_FRAMES": "-1"}, "MOTION_WARMUP_FRAMES"),
            ({"MOTION_PIXEL_THRESHOLD": "0"}, "MOTION_PIXEL_THRESHOLD"),
            ({"MOTION_PIXEL_THRESHOLD": "256"}, "MOTION_PIXEL_THRESHOLD"),
            ({"MOTION_PERSON_CONFIDENCE": "nan"}, "MOTION_PERSON_CONFIDENCE"),
            ({"MOTION_PERSON_MARGIN": "-0.1"}, "MOTION_PERSON_MARGIN"),
            ({"MOTION_PERSON_MARGIN": "nan"}, "MOTION_PERSON_MARGIN"),
            ({"MOTION_CONFIRM_FRAMES": "0"}, "MOTION_CONFIRM_FRAMES"),
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, key):
                load_config(write_env(MINIMAL), environ=values)

    def test_invalid_number_names_the_key(self):
        with self.assertRaises(ValueError) as ctx:
            load_config(write_env(MINIMAL + "DETECTION_CONFIDENCE=high\n"), environ={})
        self.assertIn("DETECTION_CONFIDENCE", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
