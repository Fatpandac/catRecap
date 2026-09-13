import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from catrecap import pipeline, recorder
from catrecap.config import Config


class RecordingWatchdogTest(unittest.TestCase):
    def test_network_timeout_and_flush_are_configured_for_segment_files(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(recorder.subprocess, "Popen") as popen:
            recorder.start_recording("rtsp://unused", Path(temp), 30)
            command = popen.call_args.args[0]
            self.assertIn("-timeout", command)
            self.assertGreater(int(command[command.index("-timeout") + 1]), 0)
            self.assertIn("-segment_format_options", command)
            self.assertIn("flush_packets=1", command[command.index("-segment_format_options") + 1])

    def test_alive_but_stalled_recorder_is_restarted_without_losing_old_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            segment = root / datetime.now().strftime("seg-%Y%m%d-%H%M%S.ts")
            segment.write_bytes(b"old footage")
            process = Mock()
            process.poll.return_value = None
            with (
                patch.object(pipeline.recorder, "start_recording", return_value=process) as start,
                patch.object(pipeline.time, "monotonic", return_value=100) as clock,
            ):
                recording = pipeline._Recording(Config("unused", "unused", "unused", segment_dir=root), "unused")
                recording.tick()
                clock.return_value = 131
                recording.tick()
                self.assertEqual(start.call_count, 2)
                process.terminate.assert_called_once()
                process.wait.assert_called_once()
                self.assertEqual(segment.read_bytes(), b"old footage")

    def test_no_recording_file_is_restarted_after_startup_grace(self):
        with tempfile.TemporaryDirectory() as temp:
            process = Mock()
            process.poll.return_value = None
            with (
                patch.object(pipeline.recorder, "start_recording", return_value=process) as start,
                patch.object(pipeline.time, "monotonic", return_value=100) as clock,
            ):
                recording = pipeline._Recording(Config("unused", "unused", "unused", segment_dir=Path(temp)), "unused")
                recording.tick()
                self.assertEqual(start.call_count, 1)
                clock.return_value = 131
                recording.tick()
                self.assertEqual(start.call_count, 2)

    def test_forced_stop_reaps_the_recorder_process(self):
        with tempfile.TemporaryDirectory() as temp:
            process = Mock()
            process.wait.side_effect = [pipeline.subprocess.TimeoutExpired("ffmpeg", 10), 0]
            with patch.object(pipeline.recorder, "start_recording", return_value=process):
                recording = pipeline._Recording(Config("unused", "unused", "unused", segment_dir=Path(temp)), "unused")
                recording.stop()
            process.kill.assert_called_once()
            self.assertEqual(process.wait.call_count, 2)

    def test_writing_segment_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            segment = root / datetime.now().strftime("seg-%Y%m%d-%H%M%S.ts")
            segment.write_bytes(b"one frame")
            process = Mock()
            process.poll.return_value = None
            with (
                patch.object(pipeline.recorder, "start_recording", return_value=process) as start,
                patch.object(pipeline.time, "monotonic", return_value=100) as clock,
            ):
                recording = pipeline._Recording(Config("unused", "unused", "unused", segment_dir=root), "unused")
                recording.tick()
                segment.write_bytes(b"one frame and another")
                clock.return_value = 131
                recording.tick()
                self.assertEqual(start.call_count, 1)
                process.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
