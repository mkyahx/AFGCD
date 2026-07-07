import os
import sys
import unittest

import numpy as np
from PIL import Image


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class PairedMaskGeometryTests(unittest.TestCase):
    def test_crop_pair_applies_same_window_to_mask(self):
        from data.paired_mask_transforms import crop_pair

        image = Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4))
        mask = Image.fromarray(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
        )

        cropped_image, cropped_mask = crop_pair(image, mask, top=1, left=1, size=2)

        self.assertEqual(np.asarray(cropped_image).tolist(), [[5, 6], [9, 10]])
        self.assertEqual(np.asarray(cropped_mask).tolist(), [[1, 1], [1, 1]])

    def test_hflip_pair_flips_image_and_mask_together(self):
        from data.paired_mask_transforms import hflip_pair

        image = Image.fromarray(
            np.array(
                [
                    [10, 20, 30],
                    [40, 50, 60],
                ],
                dtype=np.uint8,
            )
        )
        mask = Image.fromarray(
            np.array(
                [
                    [1, 0, 0],
                    [1, 0, 0],
                ],
                dtype=np.uint8,
            )
        )

        flipped_image, flipped_mask = hflip_pair(image, mask)

        self.assertEqual(np.asarray(flipped_image).tolist(), [[30, 20, 10], [60, 50, 40]])
        self.assertEqual(np.asarray(flipped_mask).tolist(), [[0, 0, 1], [0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
