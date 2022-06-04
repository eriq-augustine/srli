import os

import jpype
import jpype.imports
import jpype.types

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
CLASSPATH = os.path.join(THIS_DIR, 'jars', '*')

def ground():
    jpype.startJVM(classpath = [CLASSPATH])

    from org.linqs.psl.java import Grounding
    Grounding.test()

# TEST
if (__name__ == '__main__'):
    ground()
