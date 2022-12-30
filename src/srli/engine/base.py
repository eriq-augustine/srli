import abc
import random
import re

import srli.rule

class BaseEngine(abc.ABC):
    WEIGHT_SLACK = 0.01

    def __init__(self, relations, rules, seed = None, evaluations = [], options = {},
            normalize_weights = True,
            **kwargs):
        self._relations = relations
        self._evaluations = evaluations
        self._options = options

        if (seed is None):
            seed = random.randint(0, 2 ** 31)
        self._rng = random.Random(seed)

        self._rules = self._normalize_rules(rules, normalize_weights)

    def solve(self, **kwargs):
        raise NotImplementedError("BaseEngine.solve")

    def learn(self, **kwargs):
        raise NotImplementedError("BaseEngine.learn")

    def ground(self, **kwargs):
        raise NotImplementedError("BaseEngine.ground")

    def _normalize_rules(self, base_rules, normalize_weights):
        rules = []
        weight_sum = 0.0

        for rule in base_rules:
            if (isinstance(rule, str)):
                rule = srli.rule.Rule(rule)
            rules.append(rule)

            if (rule.is_weighted()):
                weight_sum += rule.weight()

        for relation in self._relations:
            if (relation.has_negative_prior_weight()):
                weight_sum += relation.get_negative_prior_weight()

        if (normalize_weights and (weight_sum > (1.0 + BaseEngine.WEIGHT_SLACK))):
            for rule in rules:
                if (rule.is_weighted()):
                    rule.set_weight(rule.weight() / weight_sum)

            for relation in self._relations:
                if (relation.has_negative_prior_weight()):
                    relation.set_negative_prior_weight(relation.get_negative_prior_weight() / weight_sum)

        return rules

    def _infer_variable_types(self):
        """
        Fill in any missing types using the rules as hints.
        """

        PREFIX = 'srli_type__'

        # Build an initial map of types that includes existing types and placeholders for all missing types.
        variable_types = {}
        next_id = 0

        for relation in self._relations:
            relation_types = []

            if (relation.variable_types() is None):
                for i in range(relation.arity()):
                    relation_types.append("%s%03d" % (PREFIX, next_id))
                    next_id += 1
            else:
                for variable_type in relation.variable_types():
                    if (variable_type.startswith(PREFIX)):
                        raise ValueError("Cannot infer types when existing type (%s) shares the inferred prefix ('%s')." % (variable_type, PREFIX))
                    relation_types.append(variable_type)

            variable_types[relation.name().upper()] = relation_types

        # Use each rule to merge types.
        for rule in self._rules:
            # We are just looking for arguments, so a full parse is not necessary.
            matches = re.findall(r'(\w+\([^\)]+\))', rule.text())
            if (matches is None):
                continue

            # {variable: [type, ...], ...}
            variable_map = {}

            for raw_atom in matches:
                parts = raw_atom[0:-1].split('(', 2)

                relation_name = parts[0].upper()
                variables = [variable.strip() for variable in parts[1].split(',')]

                if (relation_name not in variable_types):
                    raise ValueError("Could not find relation (%s) from rule: '%s'." % (relation_name, rule.text()))

                for i in range(len(variables)):
                    variable = variables[i]
                    variable_type = variable_types[relation_name][i]

                    if (variable not in variable_map):
                        variable_map[variable] = set()

                    variable_map[variable].add(variable_type)

            # Merge.
            for merge_types in variable_map.values():
                if (len(merge_types) == 1):
                    continue

                representative = min(merge_types)
                for relation_types in variable_types.values():
                    for i in range(len(relation_types)):
                        if (relation_types[i] in merge_types):
                            relation_types[i] = representative

        for relation in self._relations:
            relation.set_variable_types(variable_types[relation.name().upper()])
