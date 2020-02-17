import argparse
import sys

import schumt.workflow


class Trainer:
    def __init__(self):
        self.args = self.register_args(sys.argv)
        self.workflow = schumt.workflow.builder(self.args['workflow'])

    @classmethod
    def register_args(self, source):
        parser = argparse.ArgumentParser()
        parser.add_argument("--workflow", default="translation")
        args, _ = parser.parse_known_args(source)
        return args.__dict__

    def train_epoch(self):
        while self.workflow.train_step():
            continue
