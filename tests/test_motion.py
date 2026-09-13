import unittest

import numpy as np

from catrecap.motion import MotionDetector

SIZE = (240, 320, 3)


def background(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(40, 60, SIZE, dtype=np.uint8)


def with_blob(frame, x, size=30, value=240):
    frame = frame.copy()
    frame[100 : 100 + size, x : x + size] = value
    return frame


def warmed(detector, frame, times=15):
    for _ in range(times):
        detector.update(frame)
    return detector


class MotionDetectorTest(unittest.TestCase):
    def test_static_scene_is_not_moving(self):
        base = background()
        detector = warmed(MotionDetector(), base)
        self.assertFalse(detector.update(base))
        self.assertLess(detector.last_ratio, 0.001)

    def test_moving_blob_is_detected(self):
        base = background()
        detector = warmed(MotionDetector(confirm_frames=1), base)
        self.assertTrue(detector.update(with_blob(base, 50)))
        self.assertGreater(detector.last_ratio, MotionDetector.min_ratio)

    def test_stopped_object_does_not_remain_motion(self):
        base = background()
        detector = warmed(MotionDetector(confirm_frames=1), base)
        stopped = with_blob(base, 50)
        self.assertTrue(detector.update(stopped))
        for _ in range(10):
            self.assertFalse(detector.update(stopped))

    def test_excludes_person_and_departure_but_keeps_motion_elsewhere(self):
        base = background()
        detector = warmed(MotionDetector(confirm_frames=1), base)
        person = with_blob(base, 30, size=60)
        person_box = [(25 / 320, 95 / 240, 95 / 320, 165 / 240)]
        for _ in range(60):
            self.assertFalse(detector.update(person, ignored_boxes=person_box))
        # 人不再被检出的一帧，不能把原人体位置的变化当作新运动。
        self.assertFalse(detector.update(base, ignored_boxes=[]))
        cat = with_blob(person, 200)
        self.assertTrue(detector.update(cat, ignored_boxes=person_box))
        self.assertEqual(np.count_nonzero(detector._mask[100:160, 30:90]), 0)
        self.assertGreater(np.count_nonzero(detector._mask[100:130, 200:230]), 0)

    def test_person_margin_covers_nearby_shadow_without_masking_distant_motion(self):
        base = background()
        detector = warmed(MotionDetector(person_margin=0.5, confirm_frames=1), base)
        # 检测框只有身体，影子在框外；余量按框最长边的比例扩展。
        box = [(50 / 320, 100 / 240, 80 / 320, 160 / 240)]
        shadow = with_blob(base, 90, size=20)
        self.assertFalse(detector.update(shadow, ignored_boxes=box))
        self.assertTrue(detector.update(with_blob(shadow, 220), ignored_boxes=box))

    def test_pixel_threshold_can_filter_small_brightness_changes(self):
        base = background()
        detector = warmed(MotionDetector(pixel_threshold=20), base)
        changed = base.copy()
        changed[100:130, 50:80] += 10
        self.assertFalse(detector.update(changed))

    def test_prepare_and_evaluate_count_a_frame_only_once(self):
        base = background()
        detector = warmed(MotionDetector(confirm_frames=1), base)
        mask = detector.prepare(with_blob(base, 50))
        self.assertEqual(mask.shape, base.shape[:2])
        self.assertEqual(detector.frames, 16)
        self.assertTrue(detector.evaluate(required_boxes=[(0, 0, .8, 1)]))
        self.assertEqual(detector.frames, 16)

    def test_pet_regions_are_required_not_just_motion_anywhere(self):
        base = background()
        detector = warmed(MotionDetector(confirm_frames=1), base)
        moving = with_blob(base, 50)
        self.assertFalse(detector.update(moving, required_boxes=[]))
        moving = with_blob(base, 80)
        self.assertFalse(detector.update(moving, required_boxes=[(.8, 0, 1, 1)]))
        moving = with_blob(base, 110)
        self.assertTrue(detector.update(moving, required_boxes=[(.3, .35, .5, .6)]))

    def test_missing_pet_resets_consecutive_confirmation(self):
        base = background()
        detector = warmed(MotionDetector(confirm_frames=2), base)
        region = [(0, 0, .8, 1)]
        self.assertFalse(detector.update(with_blob(base, 50), required_boxes=region))
        self.assertFalse(detector.update(with_blob(base, 80), required_boxes=[]))
        self.assertEqual(detector.consecutive_frames, 0)
        self.assertFalse(detector.update(with_blob(base, 110), required_boxes=region))
        self.assertTrue(detector.update(with_blob(base, 140), required_boxes=region))

    def test_requires_consecutive_motion_samples(self):
        base = background()
        detector = warmed(MotionDetector(confirm_frames=2), base)
        self.assertFalse(detector.update(with_blob(base, 50)))
        self.assertFalse(detector.update(base))  # 一次闪动不能触发，计数要清零
        self.assertFalse(detector.update(with_blob(base, 100)))
        self.assertTrue(detector.update(with_blob(base, 130)))
        self.assertFalse(detector.update(with_blob(base, 130)))  # 停下后立即停止续期

    def test_tiny_speck_is_ignored(self):
        base = background()
        detector = warmed(MotionDetector(), base)
        self.assertFalse(detector.update(with_blob(base, 50, size=2)))

    def test_whole_frame_change_is_ignored(self):
        # 开灯、摄像头切夜视：整幅画面变了，不是宠物在动
        base = background()
        detector = warmed(MotionDetector(), base)
        self.assertFalse(detector.update(np.full(SIZE, 230, dtype=np.uint8)))
        self.assertGreater(detector.last_ratio, MotionDetector.max_ratio)

    def test_warmup_frames_never_trigger(self):
        base = background()
        detector = MotionDetector(warmup_frames=10)
        for index in range(10):
            self.assertFalse(detector.update(with_blob(base, 10 + index * 10)))


if __name__ == "__main__":
    unittest.main()
