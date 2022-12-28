import srli.engine.problog.base

class NonCollectiveProbLog(srli.engine.problog.base.BaseGroundProbLog):
    """
    An engine that tries to run non (or less) collective chunks of a ProbLog program at a time.
    This should, hoprefully, allow larger and more complex programs to be run without issues.
    """

    DEFAULT_MAX_ITERATIONS = 10
    DEFAULT_MAX_GROUND_RULES = 50

    # TODO(eriq): Stop conditions need more work.
    DEFAULT_STOP_MOVEMENT = 0.05

    def __init__(self, relations, rules,
            max_iterations = DEFAULT_MAX_ITERATIONS, max_ground_rules = DEFAULT_MAX_GROUND_RULES,
            stop_movement = DEFAULT_STOP_MOVEMENT,
            **kwargs):
        super().__init__(relations, rules, **kwargs)

        self._max_iterations = max_iterations
        self._max_ground_rules = max_ground_rules
        self._stop_movement = stop_movement

    def solve(self, **kwargs):
        atoms, ground_rules, atom_uses, sum_constraints = self._prep()

        for iteration in range(1, self._max_iterations + 1):
            movement = self._iteration(atoms, ground_rules, atom_uses, sum_constraints)

            # Normalize movement by the number of RVAs.
            movement /= float(len(atom_uses))

            print("Iteration: %d, Movement: %f" % (iteration, movement))

            if ((iteration > 1) and (movement < self._stop_movement)):
                print("Stopping Early -- Iteration: %d, Movement: %f" % (iteration, movement))
                break

        return self._create_results(atoms)

    def _iteration(self, atoms, ground_rules, atom_uses, sum_constraints):
        movement = 0.0

        atom_ids = list(atom_uses.keys())
        self._rng.shuffle(atom_ids)

        for atom_id in atom_uses:
            target_atom_ids = set([atom_id])

            # All atoms sharing a sum constraint with the main target also gets to be a target.
            for key in atoms[atom_id].sum_constraints:
                for other_atom_id in sum_constraints[key]:
                    target_atom_ids.add(other_atom_id)

            # Any other adjacent atoms are treated as observed.
            target_ground_rules = set()
            observed_atom_ids = set()

            for target_atom_id in target_atom_ids:
                for ground_rule_index in atom_uses[target_atom_id]:
                    target_ground_rules.add(ground_rule_index)
                    observed_atom_ids |= set(ground_rules[ground_rule_index].atom_ids)

            observed_atom_ids -= target_atom_ids

            program = []

            program += self._write_ground_rules(target_ground_rules, ground_rules, atoms, target_atom_ids, sum_constraints,
                    max_ground_rules = self._max_ground_rules)
            program.append('')
            program += self._write_observations(observed_atom_ids, atoms)
            program.append('')
            program += self._write_queries(target_atom_ids, atoms)

            movement += self._run("\n".join(program), target_atom_ids, atoms)

        return movement
