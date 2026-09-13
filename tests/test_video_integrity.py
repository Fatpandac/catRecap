import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from catrecap import recorder, pipeline
from catrecap.config import Config


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "需要 ffmpeg/ffprobe")
class VideoIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "source.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=128x96:rate=10", "-t", "4", "-c:v", "libx264",
            "-preset", "ultrafast", "-tune", "zerolatency", "-g", "10", "-bf", "0",
            str(cls.source),
        ], check=True, capture_output=True)
        cls.segment = cls.root / "source.ts"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(cls.source),
            "-c", "copy", "-an", str(cls.segment),
        ], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def segments(self, root, offsets):
        start = datetime(2026, 1, 1, 10).timestamp()
        for offset in offsets:
            name = datetime.fromtimestamp(start + offset).strftime("seg-%Y%m%d-%H%M%S.ts")
            shutil.copyfile(self.segment, root / name)
        return start

    def test_seek_beyond_eof_rejects_empty_mp4_even_when_ffmpeg_succeeds(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "empty.mp4"
            self.assertFalse(recorder.cut_file(self.source, 300, 309, output))
            self.assertFalse(output.exists())

    def test_subsecond_clip_is_not_extended_or_sent_as_zero_seconds(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "short.mp4"
            self.assertFalse(recorder.cut_file(self.source, 0, .5, output))
            self.assertFalse(output.exists())

    def test_valid_video_can_pass_the_telegram_boundary(self):
        with patch("requests.post") as post:
            post.return_value.ok = True
            pipeline.send_to_telegram(Config("unused", "unused", "unused"), self.source)
            post.assert_called_once()

    def test_stale_segment_is_rejected_before_cutting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            start = self.segments(root, [0])
            with patch.object(recorder, "_cut") as cut:
                self.assertFalse(recorder.cut_clip(root, start + 300, start + 309, root / "bad.mp4"))
                cut.assert_not_called()

    def test_recording_gap_is_not_concatenated_as_continuous_footage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            start = self.segments(root, [0, 30])
            with patch.object(recorder, "_cut") as cut:
                self.assertFalse(recorder.cut_clip(root, start + 1, start + 32, root / "gap.mp4"))
                cut.assert_not_called()

    def test_contiguous_segments_produce_playable_video(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            start = self.segments(root, [0, 4])
            output = root / "valid.mp4"
            self.assertTrue(recorder.cut_clip(root, start + 3, start + 5, output))
            capture = pipeline.cv2.VideoCapture(str(output))
            try:
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape[:2], (96, 128))
            finally:
                capture.release()

    def test_clamping_pre_roll_does_not_extend_the_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            start = self.segments(root, [0])
            with patch.object(recorder, "_cut", return_value=True) as cut:
                self.assertTrue(recorder.cut_clip(root, start - 3, start + 2, root / "clamped.mp4"))
                self.assertEqual(cut.call_args.args[2], 2)

    def test_telegram_boundary_rejects_header_only_mp4_without_http(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "empty.mp4"
            # FFmpeg 对越界 seek 仍返回 0，并写出 261 字节的 MP4 头。
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(self.source),
                "-ss", "300", "-t", "9", "-c", "copy", str(output),
            ], check=True, capture_output=True)
            self.assertGreater(output.stat().st_size, 0)
            with patch("requests.post") as post:
                with self.assertRaises(ValueError):
                    pipeline.send_to_telegram(Config("unused", "unused", "unused"), output)
                post.assert_not_called()
            self.assertTrue(output.exists(), "发送边界校验失败不应删除已有文件")


if __name__ == "__main__":
    unittest.main()
