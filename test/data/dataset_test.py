import unittest

import schumt.data.dataset as dataset

mockpath = '/Users/Schureed/dataset/text/iwslt/bin'


class MyTestCase(unittest.TestCase):
    def test_language_pair_dataset(self):
        def test_init():
            inst = dataset.LanguagePairDataset(mockpath, 'de', 'en', 1024)
            self.assertTrue(isinstance(inst, dataset.LanguagePairDataset))
            return inst

        def test_iter(inst):
            for i, data in enumerate(inst):
                src = data['src']
                trg = data['trg']
                self.assertEqual(src.size(0), trg.size(0))

        inst = test_init()
        test_iter(inst)


if __name__ == '__main__':
    unittest.main()
