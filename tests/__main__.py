from tests.preprocessors_test import TestBaseProcessor
import unittest

# if __name__ == '__main__':
# suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseProcessor)
suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseProcessor)
unittest.TextTestRunner(verbosity=2).run(suite)
