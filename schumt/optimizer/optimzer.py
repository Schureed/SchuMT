import abc
import argparse
import sys


class Optimizer(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.args = self.register_args(self.__intrinsic_parser(), sys.argv)
        self.grad_accumulate = self.args['grad_accumulate']
        self.lr = self.args['learning_rate']
        self.count_step = 0

    @classmethod
    def __intrinsic_parser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--learning-rate", type=float, required=True)
        parser.add_argument("--grad-accumulate", type=int, default=1)
        return parser

    @classmethod
    @abc.abstractmethod
    def register_args(cls, parser, source) -> dict:
        raise NotImplementedError

    def before_backward(self):
        if self.count_step % self.grad_accumulate == 0:
            self._clear_grad()
        self.count_step += 1

    @abc.abstractmethod
    def _clear_grad(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _update_parameters(self):
        raise NotImplementedError

    def after_backward(self):
        self.count_step += 1
        if self.count_step % self.grad_accumulate == 0:
            self._update_parameters()
