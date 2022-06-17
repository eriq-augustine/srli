import random

import srli.grounding

DEFAULT_NOISE = 0.05
LOG_MOD = 50
FLIP_MULTIPLIER = 15

HARD_WEIGHT = 1000.0

class MLN(object):
    """
    A basic implementation of MLNs with inference using MaxWalkSat.
    If unspecified, the number of flips defaults to FLIP_MULTIPLIERx the number of unobserved atoms (similar to Tuffy).
    """

    def __init__(self, relations, rules, weights = None, **kwargs):
        self._relations = relations
        self._rules = rules

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [1.0] * len(self._rules)

    def solve(self, max_flips = None, noise = DEFAULT_NOISE, seed = None, **kwargs):
        if (seed is None):
            seed = random.randint(0, 2**31)
        rng = random.Random(seed)

        raw_ground_rules = srli.grounding.ground(self._relations, self._rules)
        ground_rules, atom_grounding_map = self._process_ground_rules(raw_ground_rules)

        atom_values = {}
        for atom_index in atom_grounding_map:
            atom_values[atom_index] = rng.randint(0, 1)

        if (max_flips is None):
            max_flips = FLIP_MULTIPLIER * len(atom_values)

        total_loss = 0.0
        for ground_rule in ground_rules:
            total_loss += ground_rule.loss(atom_values)

        print("MLN Iteration 0, Loss: %f, Max Flips: %d." % (total_loss, max_flips))

        for flip in range(1, max_flips + 1):
            if (total_loss == 0.0):
                print("Full satisfaction found.")
                break

            # Pick a random unsatisfied ground rule.
            ground_rule_index = None
            while (ground_rule_index is None or ground_rules[ground_rule_index].loss(atom_values) == 0.0):
                ground_rule_index = rng.randint(0, len(ground_rules) - 1)

            # Flip a coin.
            # On heads, flip a random atom in the ground rule.
            # On tails, flip the atom that leads to the most satisfaction.
            if (rng.random() < noise):
                flip_atom_index = rng.choice(ground_rules[ground_rule_index].atoms)
                atom_values[flip_atom_index] = 1.0 - atom_values[flip_atom_index]
            else:
                flip_atom_index = None
                flip_atom_loss = None

                # Compute the possible loss for flipping each atom.
                for atom_index in ground_rules[ground_rule_index].atoms:
                    old_atom_loss = 0.0
                    for ground_rule_index in atom_grounding_map[atom_index]: 
                        old_atom_loss += ground_rules[ground_rule_index].loss(atom_values)

                    new_atom_loss = 0.0
                    atom_values[atom_index] = 1.0 - atom_values[atom_index]
                    for ground_rule_index in atom_grounding_map[atom_index]: 
                        new_atom_loss += ground_rules[ground_rule_index].loss(atom_values)
                    atom_values[atom_index] = 1.0 - atom_values[atom_index]

                    flip_delta = old_atom_loss - new_atom_loss
                    if (flip_atom_index is None or flip_delta > flip_atom_loss):
                        flip_atom_loss = flip_delta
                        flip_atom_index = atom_index

                atom_values[flip_atom_index] = 1.0 - atom_values[flip_atom_index]

            total_loss = 0.0
            for ground_rule in ground_rules:
                total_loss += ground_rule.loss(atom_values)

            if (flip % LOG_MOD == 0):
                print("MLN Iteration %d, Loss: %f." % (flip, total_loss))

        print("MLN Inference Complete. Iteration %d, Loss: %f." % (flip, total_loss))

        results = {}
        next_index = 0
        for relation in self._relations:
            next_index += len(relation.get_observed_data())

            if (not relation.has_unobserved_data()):
                continue

            data = relation.get_unobserved_data()

            values = []
            for i in range(len(data)):
                atom = data[i][0:relation.arity()] + [atom_values[next_index]]
                values.append(atom)
                next_index += 1

            results[relation] = values

        return results

    def _process_ground_rules(self, raw_ground_rules):
        """
        Take in the raw ground rules and collapse all the observed values.
        Return a mapping of grond atoms to all involved ground rules.
        """

        atom_grounding_map = {}
        ground_rules = []

        relation_counts = []
        for relation in self._relations:
            relation_counts.append(len(relation.get_observed_data()) + len(relation.get_unobserved_data()));

        for raw_ground_rule in raw_ground_rules:
            weight = self._weights[raw_ground_rule.ruleIndex]
            if (weight is None):
                weight = HARD_WEIGHT

            skip = False
            atoms = []
            coefficients = []
            constant = raw_ground_rule.constant

            for i in range(len(raw_ground_rule.atoms)):
                observed, value = self._fetch_atom(relation_counts, raw_ground_rule.atoms[i])

                # TODO(eriq): Find and skip trivial ground rules (depends on rule/evaluation type.

                if (observed):
                    coefficient = raw_ground_rule.coefficients[i]

                    if (raw_ground_rule.operator == '|'):
                        # Skip trivials.
                        if ((coefficient == 1.0 and value == 0.0) or (coefficient == -1.0 and value == 1.0)):
                            skip = True
                            break

                    constant -= (coefficient * value)
                else:
                    atoms.append(raw_ground_rule.atoms[i])
                    coefficients.append(raw_ground_rule.coefficients[i])

            if (skip):
                continue

            ground_rule = GroundRule(weight, atoms, coefficients, constant, raw_ground_rule.operator)

            ground_rule_index = len(ground_rules)
            ground_rules.append(ground_rule)

            for atom_index in atoms:
                if (atom_index not in atom_grounding_map):
                    atom_grounding_map[atom_index] = []
                atom_grounding_map[atom_index].append(ground_rule_index)

        return ground_rules, atom_grounding_map

    def _fetch_atom(self, relation_counts, index):
        # Get the relation.
        for relation_index in range(len(self._relations)):
            if (index < relation_counts[relation_index]):
                break
            index -= relation_counts[relation_index]

        relation = self._relations[relation_index]

        # Check for an observed atom.
        if (index < len(relation.get_observed_data())):
            atom_data = relation.get_observed_data()[index]

            value = 1.0
            if (len(atom_data) == relation.arity() + 1):
                value = int(float(atom_data[-1]) > 0.0)

            return True, value

        index -= len(relation.get_observed_data())

        # Get an unobserved atom_data.

        atom_data = relation.get_unobserved_data()[index]

        value = None
        if (len(atom_data) == relation.arity() + 1):
            value = int(float(atom_data[-1]) > 0.0)

        return False, value

class GroundRule(object):
    def __init__(self, weight, atoms, coefficients, constant, operator):
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
        loss = 0.0

        for i in range(len(self.atoms)):
            truth_value = atom_values[self.atoms[i]]
            # Matching the coefficient does not incur loss.
            if ((self.coefficients[i] == 1.0 and truth_value == 0.0) or (self.coefficients[i] == -1.0 and truth_value == 1.0)):
                loss = 1.0
                break

        return loss

    def _loss_arithmetic(self, atom_values):
        sum = 0.0

        for i in range(len(self.atoms)):
            sum += self.coefficients[i] * atom_values[self.atoms[i]]

        if (sum == self.constant):
            return 0.0

        return 1.0

    def __repr__(self):
        return "Weight: %f, Operator: %s, Constant: %f, Coefficients: [%s], Atoms: [%s]." % (self.weight, self.operator, self.constant, ', '.join(map(str, self.coefficients)), ', '.join(map(str, self.atoms)))
