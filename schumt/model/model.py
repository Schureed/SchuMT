import abc
import sys


class Model(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.args = self.register_args(sys.argv)

    @classmethod
    @abc.abstractmethod
    def register_args(cls, source) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def model_forward(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self):
        raise NotImplementedError

    @abc.abstractmethod
    def cuda(self):
        raise NotImplementedError
