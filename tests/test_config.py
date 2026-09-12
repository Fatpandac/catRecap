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

    def test_invalid_number_names_the_key(self):
        with self.assertRaises(ValueError) as ctx:
            load_config(write_env(MINIMAL + "DETECTION_CONFIDENCE=high\n"), environ={})
        self.assertIn("DETECTION_CONFIDENCE", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
