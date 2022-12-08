import os
import unittest

import srli.engine

class BaseTest(unittest.TestCase):
    """
    All tests need a base for standard setup and teardown.
    """

    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    EPSILON = 1e-4

    def assertClose(self, a, b):
        self.assertTrue(abs(a - b) <= self.EPSILON)
