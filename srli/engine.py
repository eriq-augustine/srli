import abc

# TEST: Re-structure the packages around the concept of engines.

class Engine(abc.ABC):
    def __init__(self, relations, rules, **kwargs):
        self._relations = relations
        self._rules = rules

    @abc.abstractmethod
    def solve(self, **kwargs):
        pass

    @abc.abstractmethod
    def learn(self, **kwargs):
        pass
