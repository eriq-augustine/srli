import srli.grounding

class MLN(object):
    def __init__(self, relations, rules, weights = None, **kwargs):
        self._relations = relations
        self._rules = rules

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [1.0] * len(self._rules)

    def solve(self, additional_config = None, **kwargs):
        # TEST
        groundRules = srli.grounding.ground(self._relations, self._rules)

        # TEST
        return {}
