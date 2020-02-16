import argparse

import schumt.model.module.functional as F
import schumt.model.module.transformer as transformer
from . import builder
from .model import Model


@builder.register('transformer')
class TransformerModel(Model):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super(TransformerModel, self).__init__()
        self.model = transformer.Transformer(
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            **self.args
        )

    @classmethod
    def register_args(cls, source):
        parser = argparse.ArgumentParser()
        parser.add_argument("--d-model", default=512)
        parser.add_argument("--dropout", default=0.1)
        parser.add_argument("--n-head", default=8)
        parser.add_argument("--num-encoder-layers", default=6)
        parser.add_argument("--num-decoder-layers", default=6)
        parser.add_argument("--d-filter", default=2048)
        args, _ = parser.parse_known_args(source)
        return args.__dict__

    def model_forward(self, src, trg):
        return self.model(src, trg, mask=F.generate_target_mask(trg.size(1)).to(trg.device))

    def parameters(self):
        return self.model.parameters()

    def print(self):
        print(self.model)

    def cuda(self, *args, **kwargs):
        return self.model.cuda(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)
