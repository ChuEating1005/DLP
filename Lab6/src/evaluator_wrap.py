import os
from pathlib import Path

import torch

from eval.evaluator import evaluation_model

_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


class Evaluator(evaluation_model):
    def __init__(self):
        prev_cwd = os.getcwd()
        os.chdir(_EVAL_DIR)
        try:
            super().__init__()
        finally:
            os.chdir(prev_cwd)

    @torch.no_grad()
    def eval(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        images = images.cuda()
        labels = labels.cuda()
        out = self.resnet18(images)
        return self.compute_acc(out.cpu(), labels.cpu())
