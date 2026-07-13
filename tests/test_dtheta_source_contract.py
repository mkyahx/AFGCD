import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class DthetaSourceContractTests(unittest.TestCase):
    def test_guided_time_keeps_original_single_query_attention_contract(self):
        source = (REPO_ROOT / "models" / "TIME_guided.py").read_text(encoding="utf-8")

        self.assertIn("self.D_sqrt = feat_dim**-0.5", source)
        self.assertIn("nn.Parameter(torch.empty(1, feat_dim))", source)
        self.assertIn("q = self.learnable_query.repeat(B, 1).unsqueeze(1)", source)
        self.assertIn("attn = q @ k.transpose(-2, -1)", source)
        self.assertIn("attn_logits = attn * self.D_sqrt", source)

        self.assertNotIn("num_heads", source)
        self.assertNotIn("head_dim", source)
        self.assertNotIn("attn_logits.mean(dim=1)", source)

    def test_dtheta_zero_preserves_original_returned_pruning_scores(self):
        source = (REPO_ROOT / "models" / "TIME_guided.py").read_text(encoding="utf-8")

        self.assertIn("attn_for_return = attn", source)
        self.assertIn("if patch_mask is not None and self.dtheta != 0.0:", source)
        self.assertIn("attn_for_return = attn + bias / self.D_sqrt", source)
        self.assertIn("return x, attn_for_return.squeeze(1)", source)


if __name__ == "__main__":
    unittest.main()
