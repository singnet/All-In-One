from __future__ import print_function
import unittest
from dataset.celeba import CelebAAlignedDataset

class TestCelebADataset(unittest.TestCase):
    def test_init(self):
        dataset = CelebAAlignedDataset("/home/mtk/datasets/img_align_celeba")