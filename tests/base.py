import os
import unittest

import srli.engine.mln
import srli.engine.psl
import srli.inference

ENGINE_PSL = srli.engine.psl.PSL
ENGINE_TUFFY = srli.inference.Tuffy
ENGINE_MLN = srli.engine.mln.NativeMLN
ENGINE_PL = srli.inference.ProbLog
ENGINES = [ENGINE_PSL, ENGINE_TUFFY, ENGINE_MLN, ENGINE_PL]

class BaseTest(unittest.TestCase):
    """
    All tests need a base for standard setup and teardown.
    """

    TEST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    EPSILON = 1e-4

    def assertClose(self, a, b):
        self.assertTrue(abs(a - b) <= self.EPSILON)
