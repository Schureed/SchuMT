import os

from . import SEPARATOR, EOL


class Dictionary:
    def __init__(self, specials=None):
        self._counter = []
        self._itos = []
        self._stoi = {}
        self.size = len(self._counter)

        if specials is not None:
            for vocab in specials:
                self.add(vocab, 0)

    def add(self, vocab, freq=1):
        if vocab not in self._stoi:
            self._stoi[vocab] = self.size
            self._itos.append(vocab)
            self._counter.append(0)
            self.size += 1

        self._counter[self._stoi[vocab]] += freq

    def stoi(self, vocab):
        if vocab in self._stoi:
            return self._stoi[vocab]
        else:
            return self.unk()

    def itos(self, index):
        return self._itos[index]

    def freq_of(self, index_or_vocab):
        if isinstance(index_or_vocab, str):
            index = self._stoi[index_or_vocab]
        else:
            index = index_or_vocab
        return self._counter[index]

    def pad(self):
        return self.stoi('<pad>')

    def eos(self):
        return self.stoi('<eos>')

    def bos(self):
        return self.stoi('<bos>')

    def unk(self):
        return self.stoi('<unk>')

    @classmethod
    def load(cls, path):
        inst = cls()
        with open(path) as fp:
            for line in fp:
                vocab, freq = line.split(SEPARATOR)
                inst.add(vocab, int(freq))
        return inst

    def save(self, addr: str):
        os.makedirs(os.path.dirname(addr), exist_ok=True)
        with open(addr, "w", buffering=True) as fp:
            for vocab, freq in zip(self._itos, self._counter):
                fp.write(vocab)
                fp.write(SEPARATOR)
                fp.write(str(freq))
                fp.write(EOL)
