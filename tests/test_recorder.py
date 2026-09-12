import unittest
from pathlib import Path

from catrecap.recorder import parse_segment_time, select_segments

# 每段 30 秒，起点分别是 10:00:00 / 10:00:30 / 10:01:00
T0 = parse_segment_time(Path("seg-20260101-100000.ts"))
SEGMENTS = [
    (Path(f"seg-{name}.ts"), T0 + offset)
    for name, offset in (("20260101-100000", 0), ("20260101-100030", 30), ("20260101-100100", 60))
]


class SelectSegmentsTest(unittest.TestCase):
    def test_event_inside_one_segment(self):
        files, offset = select_segments(SEGMENTS, T0 + 5, T0 + 20)
        self.assertEqual([f.name for f in files], ["seg-20260101-100000.ts"])
        self.assertEqual(offset, 5)

    def test_event_spanning_segments_offsets_from_first_file(self):
        files, offset = select_segments(SEGMENTS, T0 + 25, T0 + 65)
        self.assertEqual(len(files), 3)
        self.assertEqual(files[0].name, "seg-20260101-100000.ts")
        self.assertEqual(offset, 25)

    def test_event_in_last_still_growing_segment(self):
        files, offset = select_segments(SEGMENTS, T0 + 70, T0 + 90)
        self.assertEqual([f.name for f in files], ["seg-20260101-100100.ts"])
        self.assertEqual(offset, 10)

    def test_start_before_oldest_segment_is_clamped(self):
        files, offset = select_segments(SEGMENTS, T0 - 20, T0 + 10)
        self.assertEqual(files[0].name, "seg-20260101-100000.ts")
        self.assertEqual(offset, 0)

    def test_no_segment_covers_the_event(self):
        self.assertEqual(select_segments(SEGMENTS, T0 - 100, T0 - 50), ([], 0.0))
        self.assertEqual(select_segments([], T0, T0 + 10), ([], 0.0))

    def test_parse_segment_time_ignores_foreign_files(self):
        self.assertIsNone(parse_segment_time(Path("index.csv")))
        self.assertIsNone(parse_segment_time(Path("seg-not-a-time.ts")))


if __name__ == "__main__":
    unittest.main()
