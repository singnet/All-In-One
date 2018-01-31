import unittest
from preprocessors.celeba import CelebAPreprocessor
import os

class TestBaseProcessor(unittest.TestCase):
    def test_init(self):
        print "test init"
        with self.assertRaises(KeyError):
            preprocessor = CelebAPreprocessor("test1","tests/ds")
    def test_get_meta(self):
        preprocessor = CelebAPreprocessor("test","tests/ds")
        dataset = preprocessor.get_meta("tests/ds/all.pkl")
        self.assertEqual(dataset["file_location"][0],"S005_001_00000006.png")
    def test_split_dataset(self):
        preprocessor = CelebAPreprocessor("test","tests/ds")
        if os.path.exists("tests/ds/train.pkl"):
            os.remove("tests/ds/train.pkl")
        if os.path.exists("tests/ds/test.pkl"):
            os.remove("tests/ds/test.pkl")
        if os.path.exists("tests/ds/validation.pkl"):
            os.remove("tests/ds/validation.pkl")
        preprocessor = CelebAPreprocessor("test","tests/ds")
        self.assertEqual(preprocessor.dataset_type.dataset_type,5)
        self.assertEqual(preprocessor.dataset_dir,"tests/ds")
        preprocessor = CelebAPreprocessor("test","tests/ds",split=True)
        self.assertEqual(os.path.exists("tests/ds/train.pkl"),True)
        self.assertEqual(os.path.exists("tests/ds/test.pkl"),True)
        self.assertEqual(os.path.exists("tests/ds/validation.pkl"),True)



