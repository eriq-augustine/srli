import uuid

import pslpython.model
import pslpython.partition
import pslpython.predicate
import pslpython.rule

import srli.grounding

class PSL(object):
    def __init__(self, relations, rules, weights = None, squared = None, **kwargs):
        self._relations = relations
        self._rules = rules

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [1.0] * len(self._rules)

        if (squared is not None and len(squared) > 0):
            self._squared = squared
        else:
            self._squared = [True] * len(self._rules)

    def solve(self, additional_config = None, **kwargs):
        model = pslpython.model.Model(str(uuid.uuid4()))

        for relation in self._relations:
            predicate = pslpython.predicate.Predicate(relation.name(), closed = relation.is_observed(), size = relation.arity())

            if (relation.has_observed_data()):
                predicate.add_data(pslpython.partition.Partition.OBSERVATIONS, relation.get_observed_data())

            if (relation.has_unobserved_data()):
                predicate.add_data(pslpython.partition.Partition.TARGETS, relation.get_unobserved_data())

            if (relation.has_truth_data()):
                predicate.add_data(pslpython.partition.Partition.TRUTH, relation.get_truth_data())

            model.add_predicate(predicate)

        for i in range(len(self._rules)):
            rule = pslpython.rule.Rule(self._rules[i], weighted = self._weights[i] is not None,
                    weight = self._weights[i], squared = self._squared[i])
            model.add_rule(rule)

        if (additional_config is None):
            additional_config = {}

        raw_results = model.infer(psl_config = additional_config)

        results = {}
        for (predicate, data) in raw_results.items():
            results[self._find_relation(predicate.name())] = data.to_numpy().tolist()

        return results

    def _find_relation(self, name):
        for relation in self._relations:
            if (relation.name().lower() == name.lower()):
                return relation
        return None
