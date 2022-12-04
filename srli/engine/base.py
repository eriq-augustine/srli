import abc

class BaseEngine(abc.ABC):
    def __init__(self, relations, rules, **kwargs):
        self._relations = relations
        self._rules = rules

    def solve(self, **kwargs):
        raise NotImplementedError("Engine.solve")

    def learn(self, **kwargs):
        raise NotImplementedError("Engine.learn")

    def ground(self, **kwargs):
        raise NotImplementedError("Engine.ground")
