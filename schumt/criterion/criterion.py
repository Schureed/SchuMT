import abc
import argparse
import sys

import torch

from . import builder


class Criterion(abc.ABC):
    def __init__(self):
        self.args = self.register_args(sys.argv)

    @classmethod
    @abc.abstractmethod
    def register_args(cls, source) -> dict:
        return dict()

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@builder.register("label_smoothed_cross_entropy")
class LabelSmoothedCrossEntropy(Criterion):
    def __init__(self, ignore_indexes=None):
        super(LabelSmoothedCrossEntropy, self).__init__()
        if ignore_indexes is None:
            ignore_indexes = []
        self.ignore_indexes = ignore_indexes

    @classmethod
    def register_args(cls, source) -> dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("--label-smoothing", type=float, default=0.1)
        args, _ = parser.parse_known_args(source)
        return args.__dict__

    def __call__(self, pred, target):
        target = torch.nn.functional.one_hot(target, num_classes=pred.size(-1))

        pred = pred.reshape(-1, pred.size(-1))
        target = target.reshape(-1, target.size(-1))

        mask = torch.zeros_like(target)
        for i in self.ignore_indexes:
            should_masked = torch.nonzero(target[:, i])
            mask[should_masked, :] = 1.0
        target = torch.where(
            target > 0.0,
            torch.tensor(1.0 - self.args['label_smoothing']).to(target.device),
            torch.tensor(self.args['label_smoothing'] / (pred.size(-1) - 1)).to(target.device),
        )

        mask = (1.0 - mask).to(target.dtype)
        target = target * mask

        return -torch.sum(target * torch.log_softmax(pred, dim=-1), dim=-1).mean() / mask.mean()
