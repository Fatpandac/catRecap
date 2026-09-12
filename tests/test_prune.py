import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

from catrecap.recorder import list_segments, prune_segments


def make_segments(count, size, age_step=30.0, newest_age=0.0):
    """造 count 个分段，最新的那个距今 newest_age 秒，依次往前每段间隔 age_step 秒。"""
    directory = Path(tempfile.mkdtemp())
    now = time.time()
    for index in range(count):
        started = now - newest_age - age_step * index
        name = datetime.fromtimestamp(started).strftime("seg-%Y%m%d-%H%M%S.ts")
        (directory / name).write_bytes(b"x" * size)
    return directory


class PruneSegmentsTest(unittest.TestCase):
    def test_deletes_segments_older_than_retention(self):
        directory = make_segments(4, size=10, age_step=600)  # 0 / 10 / 20 / 30 分钟前
        prune_segments(directory, keep_seconds=1500, max_bytes=10**9)  # 避开 1200 秒的临界点
        self.assertEqual(len(list_segments(directory)), 3)

    def test_deletes_oldest_until_under_size_budget(self):
        directory = make_segments(5, size=100)
        prune_segments(directory, keep_seconds=10**6, max_bytes=250)
        remaining = list_segments(directory)
        self.assertEqual(len(remaining), 2)
        # 留下来的必须是最新的两段
        self.assertEqual(remaining, sorted(remaining, key=lambda item: item[1])[-2:])
        self.assertLessEqual(sum(p.stat().st_size for p, _ in remaining), 250)

    def test_keeps_newest_segment_even_if_it_alone_exceeds_budget(self):
        directory = make_segments(3, size=100)
        prune_segments(directory, keep_seconds=10**6, max_bytes=1)
        self.assertEqual(len(list_segments(directory)), 1)

    def test_low_free_disk_forces_pruning(self):
        directory = make_segments(4, size=100)
        prune_segments(directory, keep_seconds=10**6, max_bytes=10**9, min_free_bytes=10**18)
        self.assertEqual(len(list_segments(directory)), 1)

    def test_leaves_foreign_files_alone(self):
        directory = make_segments(2, size=100, age_step=600)
        (directory / "keep-me.txt").write_text("x")
        prune_segments(directory, keep_seconds=1, max_bytes=1)
        self.assertTrue((directory / "keep-me.txt").exists())


if __name__ == "__main__":
    unittest.main()
