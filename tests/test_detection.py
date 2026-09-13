import unittest
from types import SimpleNamespace

import numpy as np

from catrecap.config import Config
from catrecap.detection import motion_regions, detect_moving_pets


class MotionRegionDetectionTest(unittest.TestCase):
    def test_regions_are_bounded_and_limited(self):
        mask = np.zeros((360, 640), dtype=np.uint8)
        for y in (0, 150, 330):
            for x in (0, 280, 610):
                mask[y:y + 20, x:x + 20] = 255
        regions = motion_regions(mask)
        self.assertEqual(len(regions), 4)
        for x1, y1, x2, y2 in regions:
            self.assertTrue(0 <= x1 < x2 <= 640)
            self.assertTrue(0 <= y1 < y2 <= 360)
            self.assertEqual((x2 - x1, y2 - y1), (128, 108))

    def test_no_motion_skips_inference(self):
        model = SimpleNamespace(predict=lambda *a, **k: self.fail("不应运行推理"))
        result = detect_moving_pets(
            model, np.zeros((720, 1280, 3), np.uint8), np.zeros((360, 640), np.uint8),
            Config("unused", "unused", "unused"), [15],
        )
        self.assertEqual(result, [])

    def test_crop_boxes_map_back_to_frame_and_filter_species(self):
        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[40:60, 60:80] = 255
        result = SimpleNamespace(boxes=SimpleNamespace(
            cls=np.array([15, 0]), conf=np.array([.9, .9]),
            xyxyn=np.array([[.25, .25, .75, .75], [0, 0, 1, 1]]),
        ))
        shapes = []

        def predict(image, **kwargs):
            shapes.append(image.shape)
            return [result]

        boxes = detect_moving_pets(
            SimpleNamespace(predict=predict), np.zeros((720, 1280, 3), np.uint8), mask,
            Config("unused", "unused", "unused"), [15],
        )
        self.assertEqual(shapes, [(216, 256, 3)])
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], [(6 + 32) / 640, 27 / 360, (6 + 96) / 640, 81 / 360])


if __name__ == "__main__":
    unittest.main()
