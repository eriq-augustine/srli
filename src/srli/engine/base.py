import abc
import random

class BaseEngine(abc.ABC):
    def __init__(self, relations, rules, seed = None, options = {}, **kwargs):
        self._relations = relations
        self._rules = rules
        self._options = options

        if (seed is None):
            seed = random.randint(0, 2 ** 31)
        self._rng = random.Random(seed)

    def solve(self, **kwargs):
        raise NotImplementedError("Engine.solve")

    def learn(self, **kwargs):
        raise NotImplementedError("Engine.learn")

    def ground(self, **kwargs):
        raise NotImplementedError("Engine.ground")
