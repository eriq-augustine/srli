import string
import uuid

import pslpython.model
import pslpython.partition
import pslpython.predicate
import pslpython.rule

import srli.engine
import srli.grounding

class PSL(srli.engine.Engine):
    def __init__(self, relations, rules, weights = None, squared = None, **kwargs):
        super().__init__(relations, rules)

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [1.0] * len(self._rules)

        if (squared is not None and len(squared) > 0):
            self._squared = squared
        else:
            self._squared = [True] * len(self._rules)

    def solve(self, additional_config = None, transform_config = None, **kwargs):
        model = self._prep_model()

        if (additional_config is None):
            additional_config = {}

        raw_results = model.infer(psl_options = additional_config, transform_config = transform_config)

        results = {}
        for (predicate, data) in raw_results.items():
            results[self._find_relation(predicate.name())] = data.to_numpy().tolist()

        return results

    def learn(self, additional_config = None, transform_config = None, **kwargs):
        model = self._prep_model()

        if (additional_config is None):
            additional_config = {}

        model.learn(psl_options = additional_config, transform_config = transform_config)

        learned_rules = model.get_rules()

        # Note that additional rules were added that are not SRLi rules (like negative priors).
        for i in range(len(self._rules)):
            if (self._weights[i] is not None):
                self._weights[i] = learned_rules[i].weight()

        current_rule = len(self._rules)
        for relation in self._relations:
            if (not relation.has_negative_prior_weight()):
                continue

            relation.set_negative_prior_weight(learned_rules[current_rule].weight())
            current_rule += 1

        return self

    def _prep_model(self):
        model = pslpython.model.Model(str(uuid.uuid4()))

        for relation in self._relations:
            predicate = pslpython.predicate.Predicate(relation.name(), size = relation.arity())

            if (relation.has_observed_data()):
                predicate.add_observed_data(relation.get_observed_data())

            if (relation.has_unobserved_data()):
                predicate.add_target_data(relation.get_unobserved_data())

            if (relation.has_truth_data()):
                predicate.add_truth_data(relation.get_truth_data())

            model.add_predicate(predicate)

        for i in range(len(self._rules)):
            rule = pslpython.rule.Rule(self._rules[i], weighted = self._weights[i] is not None,
                    weight = self._weights[i], squared = self._squared[i])
            model.add_rule(rule)

        # Add in priors as rules.
        for relation in self._relations:
            if (not relation.has_negative_prior_weight()):
                continue

            arguments = ', '.join(string.ascii_uppercase[0:relation.arity()])
            rule_text = "!%s(%s)" % (relation.name(), arguments)

            rule = pslpython.rule.Rule(rule_text, weighted = True, weight = relation.get_negative_prior_weight(), squared = True)
            model.add_rule(rule)

        # Add in functional constraints.
        # TODO(eriq): There are several assumption here, e.g., hard weight, summation on last arg, etc.
        for relation in self._relations:
            if (not relation.is_functional()):
                continue

            arguments = list(string.ascii_uppercase[0:relation.arity()])
            arguments[-1] = '+' + arguments[-1]
            arguments = ', '.join(arguments)
            rule_text = "%s(%s) = 1.0" % (relation.name(), arguments)

            rule = pslpython.rule.Rule(rule_text, weighted = False)
            model.add_rule(rule)

        return model

    def _find_relation(self, name):
        for relation in self._relations:
            if (relation.name().lower() == name.lower()):
                return relation
        return None
