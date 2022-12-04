import abc

# TEST: Re-structure the packages around the concept of engines.

class Engine(abc.ABC):
    def __init__(self, relations, rules, **kwargs):
        self._relations = relations
        self._rules = rules

    @abc.abstractmethod
    def solve(self, **kwargs):
        raise NotImplementedError("Engine.solve")

    @abc.abstractmethod
    def learn(self, **kwargs):
        raise NotImplementedError("Engine.learn")

    @abc.abstractmethod
    def ground(self, **kwargs):
        raise NotImplementedError("Engine.ground")
