import abc

import srli.engine.base

class RandomEngine(srli.engine.base.BaseEngine):
    def __init__(self, relations, rules, **kwargs):
        super().__init__(relations, rules, **kwargs)

    @abc.abstractmethod
    def get_value(self):
        pass

    def solve(self, **kwargs):
        results = {}

        for relation in self._relations:
            if (not relation.has_unobserved_data()):
                continue

            data = relation.get_unobserved_data()

            values = []

            for row in data:
                value = self.get_value()
                values.append(list(row) + [value])

            results[relation] = values

        return results

    def learn(self, **kwargs):
        return self

class RandomDiscreteEngine(RandomEngine):
    def __init__(self, relations, rules, **kwargs):
        super().__init__(relations, rules, **kwargs)

    def get_value(self):
        return float(self._rng.randint(0, 1))

class RandomContinuousEngine(RandomEngine):
    def __init__(self, relations, rules, **kwargs):
        super().__init__(relations, rules, **kwargs)

    def get_value(self):
        return self._rng.random()
