import math

import srli.engine.base
import srli.engine.psl.engine

# TODO(eriq): Atoms can be missed if they are not present in any ground rules.
class DiscreteWeightedSolver(srli.engine.base.BaseEngine):
    """
    A rough implentation of a discrete logical inference engine.
    """

    HARD_WEIGHT = 1000.0
    DEFAULT_MAX_ITERATIONS = 50
    DEFAULT_MAX_RETRIES = 5

    # TODO(eriq): Stop conditions need more work.
    DEFAULT_STOP_LOSS_DELTA = 0.05
    DEFAULT_STOP_MOTION = 0.05

    def __init__(self, relations, rules,
            max_iterations = DEFAULT_MAX_ITERATIONS, max_retries = DEFAULT_MAX_RETRIES,
            stop_loss_delta = DEFAULT_STOP_LOSS_DELTA, stop_motion = DEFAULT_STOP_MOTION,
            **kwargs):
        super().__init__(relations, rules, **kwargs)

        self._max_iterations = max_iterations
        self._max_retries = max_retries
        self._stop_loss_delta = stop_loss_delta
        self._stop_motion = stop_motion

    def learn(self, **kwargs):
        engine = srli.engine.psl.engine.PSL(self._relations, self._rules, options = self._options)
        engine.learn()

        return self

    def solve(self, **kwargs):
        atoms, ground_rules, atom_uses, sum_constraints = self._prep()

        best_loss = None
        best_attempt = None
        best_values = None

        for attempt in range(1, self._max_retries + 1):
            for atom in atoms.values():
                atom.value = bool(self._rng.randint(0, 1))

            loss = self._attempt(attempt, atoms, ground_rules, atom_uses, sum_constraints)

            if ((best_loss is None) or (loss < best_loss)):
                best_loss = loss
                best_attempt = attempt
                best_values = {atom_id : atom.value for (atom_id, atom) in atoms.items()}

        print("Using values from attempt %d (loss: %f)." % (best_attempt, best_loss))

        return self._create_results(best_values, atoms)

    def _attempt(self, attempt, atoms, ground_rules, atom_uses, sum_constraints):
        previous_loss = self._loss(atoms, ground_rules, sum_constraints)
        print("Attempt: %d, Initial Loss: %f" % (attempt, previous_loss))

        for iteration in range(1, self._max_iterations + 1):
            motion = self._iteration(atoms, ground_rules, atom_uses, sum_constraints)

            loss = self._loss(atoms, ground_rules, sum_constraints)

            loss_delta = abs(loss - previous_loss)
            previous_loss = loss

            if (iteration % 5 == 0):
                print("Attempt: %d, Iteration: %d, Loss: %f, Loss Delta: %f, Motion: %f" % (attempt, iteration, loss, loss_delta, motion))

            if ((loss_delta < self._stop_loss_delta) and (motion < self._stop_motion)):
                print("Stopping Attempt -- Attempt: %d, Iteration: %d, Loss: %f, Loss Delta: %f, Motion: %f" % (attempt, iteration, loss, loss_delta, motion))
                break

        loss = self._loss(atoms, ground_rules, sum_constraints)
        print("Attempt: %d, Final Loss: %f" % (attempt, loss))

        return loss

    def _iteration(self, atoms, ground_rules, atom_uses, sum_constraints):
        atom_ids = list(atom_uses.keys())
        self._rng.shuffle(atom_ids)

        # TODO(eriq): Motion does not track constraints.
        motion = 0

        for atom_id in atom_ids:
            if (atoms[atom_id].observed):
                continue

            atom = atoms[atom_id]
            losses = {}

            initial_value = atom.value

            for value in [False, True]:
                atom.value = value
                loss = 0.0

                for ground_rule_index in atom_uses[atom_id]:
                    loss += ground_rules[ground_rule_index].loss(atoms)

                losses[value] = loss

            if (atom.relation.has_negative_prior_weight()):
                losses[True] += atom.relation.get_negative_prior_weight()

            atom.value = (losses[True] < losses[False])

            if (atom.value != initial_value):
                motion += 1

        # Consider sum constraints.
        for ((relation, args), atom_ids) in sum_constraints.items():
            best_atom_ids = None
            best_loss = None

            # Only actual atoms.
            real_atom_ids = list(atom_ids)

            # For partial functional constraints, add in a -1 index that will set all atoms to false.
            if (relation.sum_constraint().is_partial_functional()):
                atom_ids = list(atom_ids) + [-1]

            for atom_id in atom_ids:
                loss = 0.0

                # What is the loss when setting this atom to True and the rest to False.
                for other_atom_id in real_atom_ids:
                    if (atom_id == other_atom_id):
                        atoms[atom_id].value = True
                    else:
                        atoms[other_atom_id].value = False

                # Compute loss over all involved ground rules.
                for other_atom_id in real_atom_ids:
                    if (other_atom_id not in atom_uses):
                        continue

                    for ground_rule_index in atom_uses[other_atom_id]:
                        loss += ground_rules[ground_rule_index].loss(atoms)

                if (best_loss is None):
                    best_loss = loss
                    best_atom_ids = [atom_id]
                elif (math.isclose(loss, best_loss)):
                    best_atom_ids.append(atom_id)
                elif (loss < best_loss):
                    best_loss = loss
                    best_atom_ids = [atom_id]

            best_atom_id = self._rng.choice(best_atom_ids)
            for atom_id in real_atom_ids:
                if (atom_id == best_atom_id):
                    atoms[atom_id].value = True
                else:
                    atoms[atom_id].value = False

        return (motion / float(len(atom_uses)))

    def _create_results(self, atom_values, atoms):
        results = {}

        # {(atom arg, ...): atom_id, ...}
        atom_map = {tuple(atom.arguments) : atom_id for (atom_id, atom) in atoms.items()}

        for relation in self._relations:
            if (not relation.has_unobserved_data()):
                continue

            data = relation.get_unobserved_data()

            values = []

            for row in data:
                key = tuple(row)

                if ((key in atom_map) and (atom_map[key] in atoms)):
                    value = float(atom_values[atom_map[key]])
                else:
                    # An atom not participating in any used ground rules just get a default value.
                    value = float(bool(self._rng.randint(0, 1)))

                values.append(list(row) + [value])

            results[relation] = values

        return results

    def _loss(self, atoms, ground_rules, sum_constraints):
        loss = 0.0

        for ground_rule in ground_rules:
            loss += ground_rule.loss(atoms)

        for atom in atoms.values():
            if (atom.observed):
                continue

            if (atom.value and atom.relation.has_negative_prior_weight()):
                loss += atom.relation.get_negative_prior_weight()

        for ((relation, args), atom_ids) in sum_constraints.items():
            weight = relation.sum_constraint().weight
            if (weight is None):
                weight = DiscreteWeightedSolver.HARD_WEIGHT

            positive_count = 0

            for atom_id in atom_ids:
                if (atoms[atom_id].value):
                    positive_count += 1

            if (not math.isclose(positive_count, relation.sum_constraint().constant)):
                loss += weight

        return loss

    def _prep(self):
        engine = srli.engine.psl.engine.PSL(self._relations, self._rules, options = self._options)
        ground_program = engine.ground(ignore_priors = True, ignore_sum_constraint = True, get_all_atoms = True)

        relation_map = {relation.name().upper() : relation for relation in self._relations}

        atoms = {int(atom_id) : DiscreteWeightedSolver._Atom(atom, relation_map, self._rng) for (atom_id, atom) in ground_program['atoms'].items()}
        ground_rules = [self._make_rule(ground_rule) for ground_rule in ground_program['groundRules']]

        # {atom_id: [ground_rule_index, ...], ...}
        atom_uses = {}

        for ground_rule_index in range(len(ground_rules)):
            ground_rule = ground_rules[ground_rule_index]
            for atom_id in ground_rule.atom_ids:
                if (atoms[atom_id].observed):
                    continue

                if (atom_id not in atom_uses):
                    atom_uses[atom_id] = []
                atom_uses[atom_id].append(ground_rule_index)

        # {(relation, args...): [atom_id, ...], ...}
        sum_constraints = {}

        # An observation already solved these constraints.
        # {(relation, args...), ...}
        solved_sum_constraints = set()

        for (atom_id, atom) in atoms.items():
            if (not atom.relation.has_sum_constraint()):
                continue

            constraint = atom.relation.sum_constraint()

            if ((not constraint.is_functional()) and (not constraint.is_partial_functional())):
                raise ValueError("Cannot handle sum constraints that are not (partial) functional.")

            if (atom.observed and (not atom.value)):
                # An observed False in a constraint should just be ignored (and not added to the constrained atoms).
                continue

            entity_indexes = list(range(atom.relation.arity()))
            for index in constraint.label_indexes:
                entity_indexes.remove(index)

            args = tuple([atom.arguments[index] for index in entity_indexes])
            key = (atom.relation, args)

            if (atom.observed and atom.value):
                # An observed True in a constraint means that we can solve the constraint now.
                solved_sum_constraints.add(key)
                continue

            if (key not in sum_constraints):
                sum_constraints[key] = []
            sum_constraints[key].append(atom_id)

        for key in solved_sum_constraints:
            if (key not in sum_constraints):
                continue

            # All values currently in the sum constraint for this already solved key should be set to zero.
            atom_ids = sum_constraints.pop(key)
            for atom_id in atom_ids:
                atoms[atom_id].value = False
                atoms[atom_id].observed = True

        return atoms, ground_rules, atom_uses, sum_constraints

    class _LogicalRule(object):
        def __init__(self, atom_ids, coefficients, weight):
            self.atom_ids = list(atom_ids)
            self.coefficients = coefficients
            self.weight = weight

        def loss(self, atoms):
            for i in range(len(self.atom_ids)):
                value = atoms[self.atom_ids[i]].value

                if (self.coefficients[i] < 0):
                    value = not value

                if (value):
                    return 0.0

            return self.weight

    class _ArithmeticRule(object):
        def __init__(self, atom_ids, coefficients, constant, operator, weight):
            self.atom_ids = list(atom_ids)
            self.coefficients = list(coefficients)
            self.constant = constant
            self.operator = operator
            self.weight = weight

        def loss(self, atoms):
            atom_sum = 0.0

            for i in range(len(self.atom_ids)):
                atom_sum += (atoms[self.atom_ids[i]].value * self.coefficients[i])

            if (self.operator == '<'):
                satisfied = (atom_sum < self.constant)
            elif (self.operator == '<='):
                satisfied = (atom_sum <= self.constant)
            elif (self.operator == '='):
                satisfied = (math.isclose(atom_sum, self.constant))
            elif (self.operator == '>='):
                satisfied = (atom_sum >= self.constant)
            elif (self.operator == '>'):
                satisfied = (atom_sum > self.constant)
            else:
                raise ValueError("Unknown arithmetic relation operator: '%s'." % (self.operator))

            if (satisfied):
                return 0.0

            return self.weight

    def _make_rule(self, ground_info):
        weight = ground_info['weight']
        if (weight < 0.0):
            weight = DiscreteWeightedSolver.HARD_WEIGHT

        if (ground_info['operator'] == '|'):
            return DiscreteWeightedSolver._LogicalRule(ground_info['atoms'], ground_info['coefficients'], weight)

        return DiscreteWeightedSolver._ArithmeticRule(ground_info['atoms'], ground_info['coefficients'], ground_info['constant'], ground_info['operator'], weight)

    class _Atom(object):
        def __init__(self, ground_info, relation_map, rng):
            self.relation = relation_map[ground_info['predicate'].upper()]
            self.arguments = ground_info['arguments']
            self.observed = ground_info['observed']

            if (self.observed):
                self.value = bool(int(ground_info['value']))
            else:
                self.value = bool(rng.randint(0, 1))

        def __repr__(self):
            operator = '==' if self.observed else '?='
            return "%s(%s) %s %s" % (self.relation.name(), ', '.join(map(str, self.arguments)), operator, self.value)
