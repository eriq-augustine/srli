import os
import unittest

class BaseTest(unittest.TestCase):
    """
    All PSL tests need a base for standard setup and teardown.
    """

    TEST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    EPSILON = 1e-4

    def assertClose(self, a, b):
        self.assertTrue(abs(a - b) <= self.EPSILON)
