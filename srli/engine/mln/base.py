import abc

import srli.engine.base
import srli.engine.psl

HARD_WEIGHT = 1000.0

class BaseMLN(srli.engine.base.BaseEngine):
    """
    The common base for a basic implementation of MLNs with inference using MaxWalkSat.
    """

    def __init__(self, relations, rules, weights = None, **kwargs):
        super().__init__(relations, rules, **kwargs)

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [1.0] * len(self._rules)

    def solve(self, **kwargs):
        # Specifically ground with only hard constraints so arithmetic == is not turned into <= and >=.
        engine = srli.engine.psl.PSL(self._relations, self._rules,
                weights = [None] * len(self._weights), squared = [False] * len(self._weights))
        ground_program = engine.ground(ignore_priors = True, ignore_functional = True)

        ground_rules, atoms = self._process_ground_program(ground_program)

        atom_values = self.reason(ground_rules, atoms, **kwargs)

        return self._create_results(atom_values, atoms)

    @abc.abstractmethod
    def reason(self, ground_rules, atoms):
        pass

    def _get_initial_atom_value(self, atom):
        if (atom['relation'].has_negative_prior_weight()):
            return int(self._rng.random() < atom['relation'].get_negative_prior_weight())

        return self._rng.randint(0, 1)

    def _create_results(self, atom_values, atoms):
        results = {}

        # {(atom arg, ...): atom_index, ...}
        atom_map = {tuple(atom['arguments']) : atom_index for (atom_index, atom) in atoms.items()}

        for relation in self._relations:
            if (not relation.has_unobserved_data()):
                continue

            data = relation.get_unobserved_data()

            values = []

            for row in data:
                atom_index = atom_map[tuple(row)]

                if (atom_index in atom_values):
                    value = atom_values[atom_index]
                else:
                    # An atom not participating in any non-trivial rules.
                    value = self._get_initial_atom_value(atoms[atom_index])

                values.append(list(row) + [value])

            results[relation] = values

        return results

    def _process_ground_program(self, ground_program):
        """
        Take in the raw ground rules and collapse all the observed values.
        Return a mapping of grond atoms to all involved ground rules.

        Returns:
            [GroundRule, ...]
            {atomIndex: {atom info ...}, ...}
        """

        ground_atoms = {}
        ground_rules = []

        relation_map = {relation.name().upper() : relation for relation in self._relations}

        for (atom_index_str, atom_info) in ground_program['atoms'].items():
            atom_info['relation'] = relation_map[atom_info['predicate']]
            ground_atoms[int(atom_index_str)] = atom_info

        for raw_ground_rule in ground_program['groundRules']:
            rule_index = raw_ground_rule['ruleIndex']
            operator = raw_ground_rule['operator']
            weight = self._weights[rule_index]
            constant = int(raw_ground_rule['constant'])

            raw_coefficients = raw_ground_rule['coefficients']
            raw_atoms = raw_ground_rule['atoms']

            if (weight is None):
                weight = HARD_WEIGHT

            # TODO(eriq): An additional check for trivial rules can be useful here.

            # Check the atoms for observed values (which will be folded into the constant) and trivality.
            coefficients = []
            atoms = []
            skip = False

            for i in range(len(raw_atoms)):
                atom = ground_atoms[raw_atoms[i]]
                coefficient = int(raw_coefficients[i])

                if (atom['observed']):
                    value = int(atom['value'])

                    if (operator == '|'):
                        # Skip trivials.
                        if ((coefficient == 1 and value == 0) or (coefficient == -1 and value == 1)):
                            skip = True
                            break

                    constant -= (coefficient * value)
                else:
                    coefficients.append(coefficient)
                    atoms.append(raw_atoms[i])

            if (skip):
                continue

            ground_rule = GroundRule(rule_index, weight, atoms, coefficients, constant, operator)

            ground_rule_index = len(ground_rules)
            ground_rules.append(ground_rule)

        return ground_rules, ground_atoms

class GroundRule(object):
    def __init__(self, rule_index, weight, atoms, coefficients, constant, operator):
        self.rule_index = rule_index
        self.weight = weight
        self.atoms = atoms
        self.coefficients = coefficients
        self.constant = constant
        self.operator = operator

        # TODO(eriq): Standardize and support logical and arithmetic rules.
        assert operator in ['|', '='], "Unsupported rule operator: '%s'." % (operator)

    def loss(self, atom_values):
        if (self.operator == '|'):
            loss = self._loss_logical(atom_values)
        else:
            loss = self._loss_arithmetic(atom_values)

        return self.weight * loss

    def _loss_logical(self, atom_values):
        for i in range(len(self.atoms)):
            truth_value = int(atom_values[self.atoms[i]])
            coefficient = self.coefficients[i]

            # If any atom matches the coefficient, then no loss is incured.
            if ((coefficient == -1 and truth_value == 1) or (coefficient == 1 and truth_value == 0)):
                return 0.0

        return 1.0

    def _loss_arithmetic(self, atom_values):
        sum = 0.0

        for i in range(len(self.atoms)):
            sum += self.coefficients[i] * atom_values[self.atoms[i]]

        if (sum == self.constant):
            return 0.0

        return 1.0

    def __repr__(self):
        return "Weight: %f, Operator: %s, Constant: %d, Coefficients: [%s], Atoms: [%s]." % (self.weight, self.operator, self.constant, ', '.join(map(str, self.coefficients)), ', '.join(map(str, self.atoms)))
