import math

import srli.engine.mln.base

DEFAULT_MAX_TRIES = 3
DEFAULT_NOISE = 0.05
LOG_MOD = 50
FLIP_MULTIPLIER = 10

class NativeMLN(srli.engine.mln.base.BaseMLN):
    """
    A basic implementation of MLNs with inference using MaxWalkSat.
    If unspecified, the number of flips defaults to FLIP_MULTIPLIER x the number of unobserved atoms (similar to Tuffy).
    """

    def __init__(self, relations, rules, weights = None, **kwargs):
        super().__init__(relations, rules, **kwargs)

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [1.0] * len(self._rules)

    def reason(self, ground_rules, atoms, max_flips = None, max_tries = DEFAULT_MAX_TRIES, noise = DEFAULT_NOISE, **kwargs):
        atom_rule_map = self._map_atoms(ground_rules)

        if (max_flips is None):
            max_flips = FLIP_MULTIPLIER * len(atom_rule_map)

        best_atom_values = None
        best_total_loss = None
        best_attempt = None

        for attempt in range(1, max_tries + 1):
            atom_values, total_loss = self._inference_attempt(attempt, max_flips, noise, ground_rules, atoms, atom_rule_map)
            if (best_total_loss is None or total_loss < best_total_loss):
                best_total_loss = total_loss
                best_atom_values = atom_values
                best_attempt = attempt

            if (math.isclose(total_loss, 0.0)):
                break

        print("MLN Inference Complete - Best Attempt: %d, Loss: %f." % (best_attempt, best_total_loss))

        return best_atom_values

    def _inference_attempt(self, attempt, max_flips, noise, ground_rules, atoms, atom_rule_map):
        atom_values = {}
        for atom_index in atom_rule_map:
            atom_values[atom_index] = self._get_initial_atom_value(atoms[atom_index])

        total_loss = 0.0
        for ground_rule in ground_rules:
            total_loss += ground_rule.loss(atom_values)

        print("MLN Inference - Attempt: %d, Iteration 0, Loss: %f, Max Flips: %d." % (attempt, total_loss, max_flips))

        for flip in range(1, max_flips + 1):
            if (math.isclose(total_loss, 0.0)):
                print("Full satisfaction found.")
                break

            # Pick a random unsatisfied ground rule.
            ground_rule_index = None
            while (ground_rule_index is None or math.isclose(ground_rules[ground_rule_index].loss(atom_values), 0.0)):
                ground_rule_index = self._rng.randint(0, len(ground_rules) - 1)

            # Flip a coin.
            # On heads, flip a random atom in the ground rule.
            # On tails, flip the atom that leads to the most satisfaction.
            if (self._rng.random() < noise):
                flip_atom_index = self._rng.choice(ground_rules[ground_rule_index].atoms)
                atom_values[flip_atom_index] = 1.0 - atom_values[flip_atom_index]
            else:
                flip_atom_index = None
                flip_atom_loss = None

                # Compute the possible loss for flipping each atom.
                for atom_index in ground_rules[ground_rule_index].atoms:
                    old_atom_loss = 0.0
                    for ground_rule_index in atom_rule_map[atom_index]:
                        old_atom_loss += ground_rules[ground_rule_index].loss(atom_values)

                    new_atom_loss = 0.0
                    atom_values[atom_index] = 1.0 - atom_values[atom_index]
                    for ground_rule_index in atom_rule_map[atom_index]:
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

    def _map_atoms(self, ground_rules):
        atom_rule_map = {}

        for ground_rule_index in range(len(ground_rules)):
            for atom_index in ground_rules[ground_rule_index].atoms:
                if (atom_index not in atom_rule_map):
                    atom_rule_map[atom_index] = []
                atom_rule_map[atom_index].append(ground_rule_index)

        return atom_rule_map
