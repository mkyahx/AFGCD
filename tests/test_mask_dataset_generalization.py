import os
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class FakeDataset:
    def __init__(self, path):
        self.path = path
        self.target_transform = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "image", 7, 42


class MaskDatasetGeneralizationTests(unittest.TestCase):
    def test_mask_wrapper_loads_dataset_specific_relative_path(self):
        from data.mask_dataset import DatasetWithPatchMask

        with tempfile.TemporaryDirectory() as mask_root:
            os.makedirs(os.path.join(mask_root, "cars_train"), exist_ok=True)
            expected = np.ones((14, 14), dtype=np.float32)
            np.save(os.path.join(mask_root, "cars_train", "00001.npy"), expected)

            wrapped = DatasetWithPatchMask(
                FakeDataset("/data/cars/cars_train/00001.jpg"),
                mask_root=mask_root,
                path_getter=lambda dataset, idx: dataset.path,
            )

            patch_mask = wrapped._get_mask(0)

            self.assertEqual(patch_mask.shape, (14, 14))
            self.assertEqual(float(patch_mask.sum()), 196.0)

    def test_mask_wrapper_uses_all_one_mask_without_mask_root(self):
        from data.mask_dataset import DatasetWithPatchMask

        wrapped = DatasetWithPatchMask(
            FakeDataset("/data/fgvc-aircraft-2013b/data/images/0001234.jpg"),
            mask_root=None,
            path_getter=lambda dataset, idx: dataset.path,
        )

        patch_mask = wrapped._get_mask(0)

        self.assertEqual(patch_mask.shape, (14, 14))
        self.assertTrue(np.allclose(patch_mask, np.ones((14, 14), dtype=np.float32)))

    def test_alpha_entrypoint_no_longer_hardcodes_cub_only(self):
        with open(os.path.join(REPO_ROOT, "train_repro_alpha.py"), encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertNotIn("currently supports only the CUB dataset", source)
        self.assertIn("'aircraft'", source)
        self.assertIn("'scars'", source)

    def test_dtheta_entrypoint_no_longer_hardcodes_cub_only(self):
        with open(os.path.join(REPO_ROOT, "train_repro_dtheta.py"), encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertNotIn("currently supports only the CUB dataset", source)
        self.assertIn("'aircraft'", source)
        self.assertIn("'scars'", source)


if __name__ == "__main__":
    unittest.main()
