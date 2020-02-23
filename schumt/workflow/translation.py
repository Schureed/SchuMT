import argparse
import torch
import schumt
from . import builder
from .workflow import Workflow


@builder.register('translation')
class TranslationWorkflow(Workflow):
    def __init__(self):
        super(TranslationWorkflow, self).__init__()

        self.data = schumt.data.builder(self.get_data_type, **{
            "path": self.args['data'],
            "src": self.args['src'],
            "trg": self.args['trg'],
        })

        self.model = schumt.model.builder(self.args['arch'], **{
            "src_vocab_size": self.data.vocab[self.data.src].size,
            "trg_vocab_size": self.data.vocab[self.data.trg].size,
        })

        self.optimizer = schumt.optimizer.builder(self.args['optimizer'], **{
            "parameters": self.model.parameters(),
            "d_model": self.model.args['d_model'],
        })

        self.criteria = schumt.criterion.builder(self.args['criterion'], **{
            "ignore_indexes": [self.data.vocab[self.data.trg].pad()]
        })

        if not self.args['cpu']:
            self.model.cuda()

        self.data_iter = iter(self.data)

    def train_step(self, *args, **kwargs):
        try:
            _d = next(self.data_iter)
        except StopIteration:
            return False
        src = _d['src']
        trg = _d['trg']
        if not self.args['cpu']:
            src = src.cuda()
            trg = trg.cuda()
        input_trg = trg[:, : -1]
        output_trg = trg[:, 1:]
        pred = self.model.model_forward(src, input_trg, *args, **kwargs)
        loss = self.criteria(pred.reshape(-1, pred.size(-1)), output_trg.reshape(-1, ))
        self.optimizer.before_backward()
        loss.backward()
        print(loss.data)
        print(self.decode(pred[0]))
        self.optimizer.after_backward()
        return True

    def register_args(cls, source) -> dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("--cpu", action="store_true", default=False)
        parser.add_argument("--arch", required=True)
        parser.add_argument("--optimizer", required=True)
        parser.add_argument("--criterion", required=True)
        parser.add_argument("--data", required=True)
        parser.add_argument("--trg", required=True)
        parser.add_argument("--src", required=True)
        args, _ = parser.parse_known_args(source)
        return args.__dict__

    @property
    def get_data_type(self):
        return "language_pair_dataset"

    def decode(self, tensor):
        assert tensor.dim() == 2
        ret = []
        vocab = self.data.vocab[self.data.trg]
        for prob in tensor:
            ret.append(vocab.itos(torch.argmax(prob)))
        return " ".join(ret)
            
