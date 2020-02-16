import abc
import sys


class Workflow(abc.ABC):
    def __init__(self):
        self.args = self.register_args(sys.argv)

    @classmethod
    @abc.abstractmethod
    def register_args(cls, source) -> dict:
        return dict()

    @abc.abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError
