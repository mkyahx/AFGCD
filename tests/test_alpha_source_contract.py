import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class AlphaSourceContractTests(unittest.TestCase):
    def test_alpha_uses_log_probability_bias_not_raw_logit_scaling(self):
        source = (REPO_ROOT / "models" / "TIME_alpha.py").read_text(encoding="utf-8")

        self.assertIn("alpha_mask.log()", source)
        self.assertNotIn("attn_logits = attn_logits *", source)

    def test_alpha_entrypoint_uses_paired_mask_transforms(self):
        source = (REPO_ROOT / "train_repro_alpha.py").read_text(encoding="utf-8")

        self.assertIn("get_paired_mask_transform", source)
        self.assertIn("PairedMaskViewGenerator", source)
        self.assertNotIn("from data.augmentations import get_transform", source)


if __name__ == "__main__":
    unittest.main()
