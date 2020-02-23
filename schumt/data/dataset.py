import abc
import argparse
import mmap
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.tensor

from . import builder
from .dictionary import Dictionary


def _load(path):
    return mmap.mmap(os.open(path, os.O_RDONLY), 0, access=mmap.ACCESS_READ)


def _reset_fp(f):
    """
    Reset IO cursor before and after function calling.
    """

    def __f(fp, *args, **kwargs):
        fp.seek(0)
        __value = f(fp, *args, **kwargs)
        fp.seek(0)
        return __value

    return __f


class Dataset:
    def __init__(self):
        self.args = self.register_args(sys.argv)

    @abc.abstractmethod
    def __iter__(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def register_args(self, source) -> dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("--max-tokens", type=int, required=True)
        args, _ = parser.parse_known_args(source)
        return args.__dict__


@builder.register('language_pair_dataset')
class LanguagePairDataset(Dataset):

    def __init__(self,
                 path,
                 src, trg,
                 ):
        """
        Data files are named in '[train|valid].[$src_lang|$trg_lang]' in path.
        Typically dict.$src_lang and dict.$trg_lang are also need in path
        """
        super(LanguagePairDataset, self).__init__()
        self.src = src
        self.trg = trg
        self.tokens = self.args['max_tokens']

        # Load vocabs
        self.vocab = {
            src: Dictionary.load(os.path.join(path, '.'.join([src, 'dict']))),
            trg: Dictionary.load(os.path.join(path, '.'.join([trg, 'dict']))),
        }

        data = {
            'train': {
                src: {
                    "data": _load(os.path.join(path, '.'.join(['train', src, 'data']))),
                    "index": torch.load(os.path.join(path, '.'.join(['train', src, 'index']))),
                    "cur": 0,
                },
                trg: {
                    "data": _load(os.path.join(path, '.'.join(['train', trg, 'data']))),
                    "index": torch.load(os.path.join(path, '.'.join(['train', trg, 'index']))),
                    "cur": 0,
                },
            },
            'valid': {
                src: {
                    "data": _load(os.path.join(path, '.'.join(['valid', src, 'data']))),
                    "index": torch.load(os.path.join(path, '.'.join(['valid', src, 'index']))),
                    "cur": 0,
                },
                trg: {
                    "data": _load(os.path.join(path, '.'.join(['valid', trg, 'data']))),
                    "index": torch.load(os.path.join(path, '.'.join(['valid', trg, 'index']))),
                    "cur": 0,
                },
            }
        }

        self.data = data

        self.reset()

    def reset(self):
        for subset in self.data.values():
            subset[self.src]['cur'] = 0
            subset[self.trg]['cur'] = 0

    def _fetch(self):
        src = self.data['train'][self.src]
        trg = self.data['train'][self.trg]

        src_index = src['index']
        src_data = src['data']

        trg_index = trg['index']
        trg_data = trg['data']

        src_sentences = []
        trg_sentences = []

        current_tokens = 0

        cur = src['cur']

        if cur + 1 >= src_index.size(0):
            return None, None

        longest = 2
        while current_tokens < self.tokens:
            if cur + 1 >= src_index.size(0):
                break
            next_src = torch.cat([
                    torch.tensor(self.vocab[self.src].bos()).view(-1, ),
                    torch.from_numpy(np.frombuffer(src_data[src_index[cur]: src_index[cur + 1]], dtype="int64")).view(
                        -1, ),
                    torch.tensor(self.vocab[self.src].eos()).view(-1, ),
                ])
            next_trg = torch.cat([
                    torch.tensor(self.vocab[self.trg].bos()).view(-1, ),
                    torch.from_numpy(np.frombuffer(trg_data[trg_index[cur]: trg_index[cur + 1]], dtype="int64")).view(
                        -1, ),
                    torch.tensor(self.vocab[self.trg].eos()).view(-1, ),
                ])
            
            longest = max(longest, next_src.size(0), next_trg.size(0))
            if (len(src_sentences) + 1) * longest > self.tokens:
                break
            src_sentences.append(next_src)
            trg_sentences.append(next_trg)
            current_tokens = longest * len(src_sentences)
            src['cur'] += 1
            trg['cur'] += 1
            cur += 1
        src_sentences = nn.utils.rnn.pad_sequence(src_sentences, batch_first=True,
                                                  padding_value=self.vocab[self.src].pad())
        trg_sentences = nn.utils.rnn.pad_sequence(trg_sentences, batch_first=True,
                                                  padding_value=self.vocab[self.trg].pad())
        return src_sentences, trg_sentences

    def __next__(self):
        src, trg = self._fetch()
        if src is None or trg is None:
            raise StopIteration

        return {
            "src": src,
            "trg": trg,
        }

    def __iter__(self):
        self.reset()
        return self

    @classmethod
    def preprocess(cls, raw_path, bin_path, train, valid, src_suffix, trg_suffix):
        """
        export processed "raw_path/{train, valid}.{src_suffix, trg_suffix}" to bin_path
        """

        @_reset_fp
        def build_vocab(fp):
            dic = Dictionary(specials=["<bos>", "<eos>", "<pad>", "<unk>"])
            for raw in fp:
                line = raw.strip()
                for token in line.split(' '):
                    dic.add(token)
            return dic

        @_reset_fp
        def count_lines(fp):
            num = 0
            for _ in fp:
                num += 1
            return num

        @_reset_fp
        def generate_binary(fp, vocab, subset, suffix):
            bin_writer = open(os.path.join(bin_path, '.'.join([subset, suffix, 'data'])), 'wb')
            idx = [0]

            for raw in fp:
                seq = np.array([vocab.stoi(token) for token in raw.strip().split(' ')], dtype="int64").tobytes()
                bin_writer.write(seq)
                idx.append(idx[-1] + len(seq))
            torch.save(torch.tensor(idx).to(torch.int64), os.path.join(bin_path, '.'.join([subset, suffix, 'index'])))

        src_vocab = build_vocab(open(os.path.join(raw_path, '.'.join([train, src_suffix]))))
        trg_vocab = build_vocab(open(os.path.join(raw_path, '.'.join([train, trg_suffix]))))

        src_vocab.save(os.path.join(bin_path, '.'.join([src_suffix, 'dict'])))
        trg_vocab.save(os.path.join(bin_path, '.'.join([trg_suffix, 'dict'])))

        for subset in [train, valid]:
            src_file = os.path.join(raw_path, '.'.join([subset, src_suffix]))
            assert os.path.exists(src_file)
            trg_file = os.path.join(raw_path, '.'.join([subset, trg_suffix]))
            assert os.path.exists(trg_file)

            src_fp = open(src_file)
            trg_fp = open(trg_file)

            assert count_lines(src_fp) == count_lines(trg_fp)
            generate_binary(src_fp, src_vocab, subset, src_suffix)
            generate_binary(trg_fp, trg_vocab, subset, trg_suffix)
