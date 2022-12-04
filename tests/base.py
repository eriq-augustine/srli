import os
import unittest

import srli.engine.mln.native
import srli.engine.mln.pysat
import srli.engine.psl
import srli.inference

ENGINE_PSL = srli.engine.psl.PSL
ENGINE_TUFFY = srli.inference.Tuffy
ENGINE_MLN_NATIVE = srli.engine.mln.native.NativeMLN
ENGINE_MLN_PYSAT = srli.engine.mln.pysat.PySATMLN
ENGINE_PL = srli.inference.ProbLog
ENGINES = [ENGINE_PSL, ENGINE_TUFFY, ENGINE_MLN_NATIVE, ENGINE_MLN_PYSAT, ENGINE_PL]

class BaseTest(unittest.TestCase):
    """
    All tests need a base for standard setup and teardown.
    """

    TEST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    EPSILON = 1e-4

    def assertClose(self, a, b):
        self.assertTrue(abs(a - b) <= self.EPSILON)
