import unittest

from catrecap.activity import ActivityTracker


def tracker(**kw):
    defaults = dict(move_threshold=0.02, post_seconds=5.0, max_seconds=30.0, cooldown=60.0)
    return ActivityTracker(**{**defaults, **kw})


CENTER = [(0.5, 0.5)]


def moved(step):
    return [(0.5 + step, 0.5)]


class ActivityTrackerTest(unittest.TestCase):
    def test_no_pet_no_event(self):
        t = tracker()
        self.assertIsNone(t.update(0.0, []))
        self.assertIsNone(t.update(1.0, []))

    def test_still_pet_does_not_start_event(self):
        t = tracker()
        self.assertIsNone(t.update(0.0, CENTER))
        self.assertIsNone(t.update(1.0, moved(0.001)))
        self.assertFalse(t.active)

    def test_movement_starts_event_and_stillness_ends_it_after_post_seconds(self):
        t = tracker()
        t.update(0.0, CENTER)
        self.assertEqual(t.update(1.0, moved(0.2)), "start")
        self.assertTrue(t.active)
        self.assertIsNone(t.update(3.0, moved(0.2)))
        self.assertIsNone(t.update(5.9, moved(0.2)))
        self.assertEqual(t.update(6.1, moved(0.2)), "stop")
        self.assertFalse(t.active)

    def test_disappearing_pet_ends_event(self):
        t = tracker()
        t.update(0.0, CENTER)
        t.update(1.0, moved(0.2))
        self.assertIsNone(t.update(2.0, []))
        self.assertEqual(t.update(6.5, []), "stop")

    def test_continuous_movement_stops_at_max_seconds(self):
        t = tracker()
        t.update(0.0, CENTER)
        t.update(1.0, moved(0.2))
        ts, step, result = 1.0, 0.2, None
        while result is None and ts < 60.0:
            ts += 1.0
            step = -step
            result = t.update(ts, moved(step))
        self.assertEqual(result, "stop")
        self.assertEqual(ts, 31.0)

    def test_cooldown_blocks_restart_until_elapsed(self):
        t = tracker()
        t.update(0.0, CENTER)
        t.update(1.0, moved(0.2))
        t.update(6.1, moved(0.2))  # stop at t=6.1
        self.assertIsNone(t.update(30.0, moved(-0.2)))
        self.assertFalse(t.active)
        self.assertEqual(t.update(66.2, moved(0.2)), "start")

    def test_exposes_last_move_for_debugging(self):
        t = tracker()
        t.update(0.0, CENTER)
        self.assertEqual(t.last_move, 0.0)  # 没有上一帧，无从比较
        t.update(1.0, moved(0.2))
        self.assertAlmostEqual(t.last_move, 0.2)
        t.update(2.0, [])
        self.assertEqual(t.last_move, 0.0)  # 画面里没宠物

    def test_movement_uses_centroid_of_all_pets(self):
        t = tracker()
        self.assertIsNone(t.update(0.0, [(0.2, 0.5), (0.8, 0.5)]))
        # 两只宠物对调位置，质心不变，不算活动
        self.assertIsNone(t.update(1.0, [(0.8, 0.5), (0.2, 0.5)]))


if __name__ == "__main__":
    unittest.main()
