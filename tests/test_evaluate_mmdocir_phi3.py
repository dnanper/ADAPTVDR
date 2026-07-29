import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

peft = types.ModuleType("peft")
peft.PeftModel = object
sys.modules.setdefault("peft", peft)

from evaluate.evaluate_mmdocir_phi3 import recall_at_k


class TestEvaluateMMDocIRPhi3(unittest.TestCase):
    def test_recall_at_k_counts_each_relevant_page(self):
        self.assertEqual(recall_at_k([18, 20, 4], {18, 19}, 3), 0.5)
