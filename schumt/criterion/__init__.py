import importlib
import os

import schumt.builder

builder = schumt.builder.Builder()

for _filename in os.listdir(os.path.dirname(__file__)):
    if not _filename.endswith('.py') or '__' in _filename:
        continue
    importlib.import_module(__package__ + '.' + _filename[:-3])
