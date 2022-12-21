import problog

import srli.engine.base
import srli.engine.psl.engine

class NonCollectiveProbLog(srli.engine.base.BaseEngine):
    """
    An engine that tries to run non (or less) collective chunks of a ProbLog program at a time.
    This should, hoprefully, allow larger and more complex programs to be run without issues.
    """

    DEFAULT_MAX_ITERATIONS = 10
    DEFAULT_MAX_GROUND_RULES = 50

    def __init__(self, relations, rules,
            max_iterations = DEFAULT_MAX_ITERATIONS, max_ground_rules = DEFAULT_MAX_GROUND_RULES,
            **kwargs):
        super().__init__(relations, rules, **kwargs)

        self._max_iterations = max_iterations
        self._max_ground_rules = max_ground_rules

    def learn(self, **kwargs):
        engine = srli.engine.psl.engine.PSL(self._relations, self._rules)
        engine.learn()

        return self

    def solve(self, **kwargs):
        atoms, ground_rules, atom_uses, sum_constraints = self._prep()

        for iteration in range(1, self._max_iterations + 1):
            movement = self._iteration(atoms, ground_rules, atom_uses, sum_constraints)
            print("Iteration: %d, Movement: %f" % (iteration, movement))

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

            program += self._write_ground_rules(target_ground_rules, ground_rules, atoms, target_atom_ids, sum_constraints)
            program.append('')
            program += self._write_observations(observed_atom_ids, atoms)
            program.append('')
            program += self._write_queries(target_atom_ids, atoms)

            # print("\n".join(program))

            movement += self._run("\n".join(program), target_atom_ids, atoms)

        return movement

    def _run(self, text, query_atom_ids, atoms):
        program = problog.program.PrologString(text)

        raw_results = problog.get_evaluatable().create_from(program).evaluate()
        raw_results = {str(key): float(value) for (key, value) in raw_results.items()}

        movement = 0.0

        # {atom_str: atom_id, ...}
        query_map = {atoms[atom_id].to_problog() : atom_id for atom_id in query_atom_ids}

        for (atom_str, value) in raw_results.items():
            if (atom_str not in query_map):
                raise ValueError("Could not locate query result (%s), queries: (%s)." % (atom_str, ', '.join(query_map.keys())))

            movement += abs(atoms[query_map[atom_str]].value - value)
            atoms[query_map[atom_str]].value = value

        return movement

    def _write_queries(self, query_atom_ids, atoms):
        program = ['% Queries', '']

        for query_atom_id in query_atom_ids:
            program.append("query(%s) ." % (atoms[query_atom_id].to_problog()))

        return program

    def _write_observations(self, observed_atom_ids, atoms):
        program = ['% Observations', '']

        for observed_atom_id in observed_atom_ids:
            program.append("%f :: %s ." % (atoms[observed_atom_id].value, atoms[observed_atom_id].to_problog()))

        return program

    def _write_ground_rules(self, ground_rule_ids, ground_rules, atoms, query_atom_ids, sum_constraints):
        program = ['% Rules', '']

        if (len(ground_rule_ids) > self._max_ground_rules):
            ground_rule_ids = self._rng.choices(list(ground_rule_ids), k = self._max_ground_rules)

        # Write normal ground rules.
        for ground_rule_id in ground_rule_ids:
            program.append(ground_rules[ground_rule_id].to_problog(atoms, query_atom_ids, self._rng))

        # Wite any annotated disjunctions.

        # Make sure not to write atoms more than once for the same sum constraint.
        # {(sum_key, atom_id), ...}
        seen_sum_atoms = set()

        for query_atom_id in query_atom_ids:
            if (len(atoms[query_atom_id].sum_constraints) == 0):
                continue

            for sum_key in atoms[query_atom_id].sum_constraints:
                key = (sum_key, query_atom_id)
                if (key in seen_sum_atoms):
                    continue

                sum_atom_ids = sum_constraints[sum_key]
                for sum_atom_id in sum_atom_ids:
                    key = (sum_key, sum_atom_id)
                    seen_sum_atoms.add(key)

                disjunction = ["1/%d :: %s" % (len(sum_atom_ids), atoms[sum_atom_id].to_problog()) for sum_atom_id in sum_atom_ids]
                program.append("%s ." % (' ; '.join(disjunction)))

        return program

    def _create_results(self, atoms):
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
                    value = float(atoms[atom_map[key]].value)
                else:
                    # An atom not participating in any used ground rules just get a default value.
                    value = float(self._rng.randint(0, 1))

                values.append(list(row) + [value])

            results[relation] = values

        return results

    def _prep(self):
        engine = srli.engine.psl.engine.PSL(self._relations, self._rules)
        ground_program = engine.ground(ignore_priors = True, ignore_sum_constraint = True)

        relation_map = {relation.name().upper() : relation for relation in self._relations}

        atoms = {int(atom_id) : NonCollectiveProbLog._Atom(atom, relation_map, self._rng) for (atom_id, atom) in ground_program['atoms'].items()}
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

            if (not constraint.is_functional()):
                raise ValueError("Cannot handle sum constraints that are not functional.")

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
            if (key in sum_constraints):
                sum_constraints.pop(key)

        for (key, atom_ids) in sum_constraints.items():
            for atom_id in atom_ids:
                atoms[atom_id].sum_constraints.append(key)

        return atoms, ground_rules, atom_uses, sum_constraints

    class _LogicalRule(object):
        def __init__(self, atom_ids, coefficients, weight):
            self.atom_ids = list(atom_ids)
            self.coefficients = coefficients
            self.weight = weight

        # TODO(eriq): Weight is not included.
        def to_problog(self, atoms, queries, rng):
            possible_heads = set(self.atom_ids) & set(queries)

            if (len(possible_heads) == 0):
                raise ValueError("No possible heads found for rule: '%s'." % (self))

            possible_heads = list(sorted(possible_heads))
            rng.shuffle(possible_heads)

            head_atom_id = None
            if (len(possible_heads) > 1):
                # Favor positive heads.
                for possible_head in possible_heads:
                    if (self.coefficients[self.atom_ids.index(possible_head)] > 0.0):
                        head_atom_id = possible_head
                        break

            if (head_atom_id is None):
                head_atom_id = possible_heads[0]

            head_atom = atoms[head_atom_id]
            body_atoms = [atoms[atom_id] for atom_id in (set(self.atom_ids) - set([head_atom_id]))]

            return "1.0 :: %s :- %s ." % (head_atom.to_problog(), ', '.join([atom.to_problog() for atom in body_atoms]))

        def __repr__(self):
            return ' | '.join([str(int(self.coefficients[i]) * self.atom_ids[i]) for i in range(len(self.atom_ids))])


    class _ArithmeticRule(object):
        def __init__(self, atom_ids, coefficients, constant, operator, weight):
            self.atom_ids = list(atom_ids)
            self.coefficients = list(coefficients)
            self.constant = constant
            self.operator = operator
            self.weight = weight

            raise NotImplementedError("Arithmetic Rules")

    def _make_rule(self, ground_info):
        weight = ground_info['weight']
        if (weight < 0.0):
            weight = NonCollectiveProbLog.HARD_WEIGHT

        if (ground_info['operator'] == '|'):
            return NonCollectiveProbLog._LogicalRule(ground_info['atoms'], ground_info['coefficients'], weight)

        return NonCollectiveProbLog._ArithmeticRule(ground_info['atoms'], ground_info['coefficients'], ground_info['constant'], ground_info['operator'], weight)

    class _Atom(object):
        def __init__(self, ground_info, relation_map, rng):
            self.relation = relation_map[ground_info['predicate'].upper()]
            self.arguments = ground_info['arguments']
            self.observed = ground_info['observed']

            # Involved sum constraints.
            self.sum_constraints = []

            if (self.observed):
                self.value = float(ground_info['value'])
            else:
                self.value = float(rng.randint(0, 1))

        def to_problog(self):
            return "%s(%s)" % (self.relation.name().lower(), ','.join(map(lambda x: str(x).lower(), self.arguments)))

        def __repr__(self):
            operator = '==' if self.observed else '?='
            return "%s(%s) %s %s" % (self.relation.name(), ', '.join(map(str, self.arguments)), operator, self.value)
