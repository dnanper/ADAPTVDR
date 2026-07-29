import sys
import unittest
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from train.schedule_utils import make_scheduler


class TestPhi3ResumeScheduler(unittest.TestCase):
    def test_resume_scheduler_matches_completed_steps(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        uninterrupted_optimizer = torch.optim.SGD([parameter], lr=3e-5)
        uninterrupted = get_cosine_schedule_with_warmup(uninterrupted_optimizer, 6, 255)
        for _ in range(200):
            uninterrupted_optimizer.step()
            uninterrupted.step()

        resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
        resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=3e-5)
        resumed = make_scheduler(resumed_optimizer, warmup_steps=6, total_steps=255, completed_steps=200)

        self.assertEqual(resumed.last_epoch, 200)
        self.assertAlmostEqual(resumed.get_last_lr()[0], uninterrupted.get_last_lr()[0])
