import string
import uuid

import pslpython.model
import pslpython.partition
import pslpython.predicate
import pslpython.rule

import srli.engine.base

class PSL(srli.engine.base.BaseEngine):
    def __init__(self, relations, rules, weights = None, squared = None, **kwargs):
        super().__init__(relations, rules, **kwargs)

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [1.0] * len(self._rules)

        if (squared is not None and len(squared) > 0):
            self._squared = squared
        else:
            self._squared = [True] * len(self._rules)

    def solve(self, additional_config = {}, transform_config = None, **kwargs):
        model = self._prep_model(additional_config = additional_config)

        raw_results = model.infer(transform_config = transform_config)

        results = {}
        for (predicate, data) in raw_results.items():
            results[self._find_relation(predicate.name())] = data.to_numpy().tolist()

        return results

    def learn(self, additional_config = {}, transform_config = None, **kwargs):
        model = self._prep_model(additional_config)

        model.learn(transform_config = transform_config)

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

    def ground(self, additional_config = {}, ignore_priors = False, ignore_functional = False, transform_config = None, **kwargs):
        model = self._prep_model(additional_config, ignore_priors, ignore_functional)
        return model.ground(transform_config = transform_config)

    def _prep_model(self, additional_config = {}, ignore_priors = False, ignore_functional = False):
        model = pslpython.model.Model(str(uuid.uuid4()))

        options = dict(self._options)
        options.update(additional_config)
        model.add_options(options)

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
        if (not ignore_priors):
            for relation in self._relations:
                if (not relation.has_negative_prior_weight()):
                    continue

                arguments = ', '.join(string.ascii_uppercase[0:relation.arity()])
                rule_text = "!%s(%s)" % (relation.name(), arguments)

                rule = pslpython.rule.Rule(rule_text, weighted = True, weight = relation.get_negative_prior_weight(), squared = True)
                model.add_rule(rule)

        # Add in functional constraints.
        # TODO(eriq): There are several assumption here, e.g., hard weight, summation on last arg, etc.
        if (not ignore_functional):
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
