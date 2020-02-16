import argparse

from schumt.workflow import builder
from .workflow import Workflow


@builder.register('translation')
class TranslationWorkflow(Workflow):
    def __init__(self, optimizer, model, data, criteria):
        super(TranslationWorkflow, self).__init__()
        self.model = model
        if not self.args['cpu']:
            self.model.cuda()
        self.optimizer = optimizer
        self.data_iter = iter(data)
        self.criteria = criteria

    def train_step(self, *args, **kwargs):
        try:
            _d = next(self.data_iter)
        except StopIteration:
            return False
        src = _d['src']
        trg = _d['trg']
        if not self.args['cpu']:
            src.cuda()
            trg.cuda()
        input_trg = trg[:, : -1]
        output_trg = trg[:, 1:]
        pred = self.model.model_forward(src, input_trg, *args, **kwargs)
        loss = self.criteria(pred.reshape(-1, pred.size(-1)), output_trg.reshape(-1, ))
        self.optimizer.zero_grad()
        loss.backward()
        print(loss.data)
        self.optimizer.step()
        return True

    def register_args(cls, source) -> dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("--cpu", action="store_true", default=False)
        args, _ = parser.parse_known_args(source)
        return args.__dict__
