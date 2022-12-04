import random

import srli.engine.base
import srli.engine.psl

DEFAULT_MAX_TRIES = 3
DEFAULT_NOISE = 0.05
LOG_MOD = 50
FLIP_MULTIPLIER = 10

HARD_WEIGHT = 1000.0

# TODO(eriq): The weight for negative priors is not handled consistently, i.e., it is treated both as a weight and probability.
# TODO(eriq): The prior should also be taken into account when flipping. The above issue makes this hard.

class NativeMLN(srli.engine.base.BaseEngine):
    """
    A basic implementation of MLNs with inference using MaxWalkSat.
    If unspecified, the number of flips defaults to FLIP_MULTIPLIER x the number of unobserved atoms (similar to Tuffy).
    """

    def __init__(self, relations, rules, weights = None, **kwargs):
        super().__init__(relations, rules)

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [1.0] * len(self._rules)

    def solve(self, max_flips = None, max_tries = DEFAULT_MAX_TRIES, noise = DEFAULT_NOISE, seed = None, **kwargs):
        if (seed is None):
            seed = random.randint(0, 2 ** 31)
        rng = random.Random(seed)

        # Specifically ground with only hard constraints so arithmetic == is not turned into <= and >=.
        engine = srli.engine.psl.PSL(self._relations, self._rules,
                weights = [None] * len(self._weights), squared = [False] * len(self._weights))
        ground_program = engine.ground(ignore_priors = True, ignore_functional = True)

        ground_rules, atoms, atom_grounding_map = self._process_ground_program(ground_program)

        if (max_flips is None):
            max_flips = FLIP_MULTIPLIER * len(atom_grounding_map)

        best_atom_values = None
        best_total_loss = None
        best_attempt = None

        for attempt in range(1, max_tries + 1):
            atom_values, total_loss = self._inference_attempt(attempt, max_flips, rng, noise, ground_rules, atoms, atom_grounding_map)
            if (best_total_loss is None or total_loss < best_total_loss):
                best_total_loss = total_loss
                best_atom_values = atom_values
                best_attempt = attempt

            if (total_loss == 0.0):
                break

        print("MLN Inference Complete - Best Attempt: %d, Loss: %f." % (best_attempt, best_total_loss))

        return self._create_results(best_atom_values, atoms, rng)

    def _get_initial_atom_value(self, rng, atom):
        if (atom['relation'].has_negative_prior_weight()):
            return int(rng.random() < atom['relation'].get_negative_prior_weight())

        return rng.randint(0, 1)

    def _inference_attempt(self, attempt, max_flips, rng, noise, ground_rules, atoms, atom_grounding_map):
        atom_values = {}
        for atom_index in atom_grounding_map:
            atom_values[atom_index] = self._get_initial_atom_value(rng, atoms[atom_index])

        total_loss = 0.0
        for ground_rule in ground_rules:
            total_loss += ground_rule.loss(atom_values)

        print("MLN Inference - Attempt: %d, Iteration 0, Loss: %f, Max Flips: %d." % (attempt, total_loss, max_flips))

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
                print("MLN Inference - Attempt: %d, Iteration %d, Loss: %f." % (attempt, flip, total_loss))

        print("MLN Inference Attempt Complete - Attempt: %d, Iteration %d, Loss: %f." % (attempt, flip, total_loss))

        return atom_values, total_loss

    def _create_results(self, atom_values, atoms, rng):
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
                    value = self._get_initial_atom_value(rng, atoms[atom_index])

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
            {atom_index: [ground_rule_index, ...], ...}
        """

        atom_grounding_map = {}
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
            constant = raw_ground_rule['constant']

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
                coefficient = raw_coefficients[i]

                if (atom['observed']):
                    value = atom['value']

                    if (operator == '|'):
                        # Skip trivials.
                        if ((coefficient == 1.0 and value == 0.0) or (coefficient == -1.0 and value == 1.0)):
                            skip = True
                            break

                    constant -= (coefficient * value)
                else:
                    coefficients.append(coefficient)
                    atoms.append(raw_atoms[i])

            if (skip):
                continue

            ground_rule = GroundRule(weight, atoms, coefficients, constant, operator)

            ground_rule_index = len(ground_rules)
            ground_rules.append(ground_rule)

            for atom_index in atoms:
                if (atom_index not in atom_grounding_map):
                    atom_grounding_map[atom_index] = []
                atom_grounding_map[atom_index].append(ground_rule_index)

        return ground_rules, ground_atoms, atom_grounding_map

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
        for i in range(len(self.atoms)):
            truth_value = atom_values[self.atoms[i]]
            coefficient = self.coefficients[i]

            # If any atom matches the coefficient, then no loss is incured.
            if ((coefficient == -1.0 and truth_value == 1.0) or (coefficient == 1.0 and truth_value == 0.0)):
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
        return "Weight: %f, Operator: %s, Constant: %f, Coefficients: [%s], Atoms: [%s]." % (self.weight, self.operator, self.constant, ', '.join(map(str, self.coefficients)), ', '.join(map(str, self.atoms)))
