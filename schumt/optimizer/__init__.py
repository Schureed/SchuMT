import importlib
import os

from schumt.builder import Builder

builder = Builder()

for filename in os.listdir(os.path.dirname(__file__)):
    if not filename.endswith('.py') or '__' in filename:
        continue
    importlib.import_module(__package__ + '.' + filename[:-3])
