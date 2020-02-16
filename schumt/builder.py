class Builder:
    """
    usage: paste the following code to __init__.py
-----------------------------------------------------------------------
import importlib
import os

from schumt.builder import Builder

builder = Builder()

for filename in os.listdir(os.path.dirname(__file__)):
    if not filename.endswith('.py') or '__' in filename:
        continue
    importlib.import_module(__package__ + '.' + filename[:-3])

-----------------------------------------------------------------------
    """

    def __init__(self):
        self.__all__ = {}

    def register(self, name):
        def __register(cls):
            assert name not in self.__all__.keys()
            assert cls not in self.__all__.items()
            self.__all__[name] = cls
            return cls

        return __register

    def build(self, name, *args, **kwargs):
        assert name in self.__all__
        return self.__all__[name](*args, **kwargs)

    def __call__(self, name, *args, **kwargs):
        return self.build(name, *args, **kwargs)
