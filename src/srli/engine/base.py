import abc
import random

import srli.rule

class BaseEngine(abc.ABC):
    WEIGHT_SLACK = 0.01

    def __init__(self, relations, rules, seed = None, evaluations = [], options = {},
            noramlize_weights = True,
            **kwargs):
        self._relations = relations
        self._evaluations = evaluations
        self._options = options

        if (seed is None):
            seed = random.randint(0, 2 ** 31)
        self._rng = random.Random(seed)

        self._rules = self._normalize_rules(rules, noramlize_weights)

    def _normalize_rules(self, base_rules, normalize_weights):
        rules = []
        weight_sum = 0.0

        for rule in base_rules:
            if (isinstance(rule, str)):
                rule = srli.rule.Rule(rule)
            rules.append(rule)

            if (rule.is_weighted()):
                weight_sum += rule.weight()

        if (normalize_weights and (weight_sum > (1.0 + BaseEngine.WEIGHT_SLACK))):
            for rule in rules:
                if (rule.is_weighted()):
                    rule.set_weight(rule.weight() / weight_sum)

        return rules

    def solve(self, **kwargs):
        raise NotImplementedError("Engine.solve")

    def learn(self, **kwargs):
        raise NotImplementedError("Engine.learn")

    def ground(self, **kwargs):
        raise NotImplementedError("Engine.ground")
