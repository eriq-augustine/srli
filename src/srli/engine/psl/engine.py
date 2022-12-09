import string
import uuid

import pslpython.model
import pslpython.partition
import pslpython.predicate
import pslpython.rule

import srli.engine.base
import srli.evaluation

class PSL(srli.engine.base.BaseEngine):
    EVAL_MAP = {
        srli.evaluation.CategoricalAccuracy: 'CategoricalEvaluator',
        srli.evaluation.F1: 'DiscreteEvaluator',
        srli.evaluation.RMSE: 'ContinuousEvaluator',
        srli.evaluation.AuROC: 'AUCEvaluator',
        srli.evaluation.AuPRC: 'AUCEvaluator',
    }

    def __init__(self, relations, rules, **kwargs):
        super().__init__(relations, rules, **kwargs)

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
            if (self._rules[i].is_weighted()):
                self._rules[i].set_weight(learned_rules[i].weight())

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

        # {relation: [eval, ...], ...}
        evals_map = {}
        for evaluation in self._evaluations:
            if (evaluation.relation() not in evals_map):
                evals_map[evaluation.relation()] = []
            evals_map[evaluation.relation()].append(evaluation)

        for relation in self._relations:
            evaluations = []

            if (relation in evals_map):
                for base_evaluation in evals_map[relation]:
                    evaluation = self._convert_evaluation(base_evaluation)
                    if (evaluation is not None):
                        evaluations.append(evaluation)

            predicate = pslpython.predicate.Predicate(relation.name(), size = relation.arity(), evaluations = evaluations)

            if (relation.has_observed_data()):
                predicate.add_observed_data(relation.get_observed_data())

            if (relation.has_unobserved_data()):
                predicate.add_target_data(relation.get_unobserved_data())

            if (relation.has_truth_data()):
                predicate.add_truth_data(relation.get_truth_data())

            model.add_predicate(predicate)

        for i in range(len(self._rules)):
            rule = pslpython.rule.Rule(self._rules[i].text(),
                    weighted = self._rules[i].is_weighted(), weight = self._rules[i].weight(),
                    squared = self._rules[i].options().get('squared', False))
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

    def _convert_evaluation(self, evaluation):
        if (type(evaluation) not in PSL.EVAL_MAP):
            return None

        return {
            'evaluator': PSL.EVAL_MAP[type(evaluation)],
            'options': evaluation.options(),
            'primary': evaluation.is_primary(),
        }

    def _find_relation(self, name):
        for relation in self._relations:
            if (relation.name().lower() == name.lower()):
                return relation
        return None
