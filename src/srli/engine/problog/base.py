import math

import problog

import srli.engine.base
import srli.engine.psl.engine

class BaseGroundProbLog(srli.engine.base.BaseEngine):
    """
    An abstract engine base that works with pre-grounding programs.
    """

    HARD_WEIGHT = 1000.0

    def __init__(self, relations, rules, **kwargs):
        super().__init__(relations, rules, **kwargs)

    def learn(self, **kwargs):
        engine = srli.engine.psl.engine.PSL(self._relations, self._rules, options = self._options)
        engine.learn()

        return self

    def _run(self, text, query_atom_ids, atoms):
        # print(text)

        try:
            program = problog.program.PrologString(text)
            problog_program = problog.get_evaluatable().create_from(program)
            raw_results = problog_program.evaluate()
        except Exception as ex:
            print("Failed to run ProbLog program:")
            print('---')
            print(text)
            print('---')
            raise ex

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

    def _write_ground_rules(self, ground_rule_ids, ground_rules, atoms, query_atom_ids, sum_constraints, max_ground_rules = -1):
        program = ['% Rules', '']

        if ((max_ground_rules > 0) and (len(ground_rule_ids) > self._max_ground_rules)):
            ground_rule_ids = self._rng.choices(list(ground_rule_ids), k = self._max_ground_rules)

        # Write normal ground rules.
        for ground_rule_id in ground_rule_ids:
            program.append(ground_rules[ground_rule_id].to_problog(atoms, query_atom_ids, self._rng))

        # Use annotated disjunctions to represent summation constraints.

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

                # For full functional constraints, evenly split the probability across all the options.
                # For partial functional constraints, add in the null option by adding one to the denominator.
                # https://problog.readthedocs.io/en/latest/modeling_basic.html#annotated-disjunctions
                denom = len(sum_atom_ids)
                if (atoms[query_atom_id].relation.sum_constraint().is_partial_functional()):
                    denom += 1

                disjunction = ["1/%d :: %s" % (denom, atoms[sum_atom_id].to_problog()) for sum_atom_id in sum_atom_ids]
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
        engine = srli.engine.psl.engine.PSL(self._relations, self._rules, options = self._options)
        ground_program = engine.ground(ignore_priors = True, ignore_sum_constraint = True, get_all_atoms = True)

        relation_map = {relation.name().upper() : relation for relation in self._relations}

        atoms = {int(atom_id) : BaseGroundProbLog._Atom(atom, relation_map, self._rng) for (atom_id, atom) in ground_program['atoms'].items()}
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

        # Check for sum constraints.
        # Check if they can be solved now, and index the unsolved ones.

        # {(relation, args...): [atom_id, ...], ...}
        sum_constraints = {}

        # A positive observation indicates that on of these constraints is already solved.
        # {(relation, args...), ...}
        solved_sum_constraints = set()

        for (atom_id, atom) in atoms.items():
            if (not atom.relation.has_sum_constraint()):
                continue

            constraint = atom.relation.sum_constraint()
            if ((not constraint.is_functional()) and (not constraint.is_partial_functional())):
                raise ValueError("Cannot handle sum constraints that are not (partial) functional.")

            if (atom.observed and math.isclose(atom.value, 0.0)):
                # An observed False in a constraint should just be ignored (and not added to the constrained atoms).
                continue

            entity_indexes = list(range(atom.relation.arity()))
            for index in constraint.label_indexes:
                entity_indexes.remove(index)

            args = tuple([atom.arguments[index] for index in entity_indexes])
            key = (atom.relation, args)

            if (atom.observed and math.isclose(atom.value, 1.0)):
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
                atoms[atom_id].value = 0.0
                atoms[atom_id].observed = True

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
        TYPE_BINARY_EQUALITY = 'binary_equality'
        TYPE_FIXED_BINARY_VALUE = 'fixed_binary_value'

        def __init__(self, atom_ids, coefficients, constant, operator, weight):
            self.atom_ids = list(atom_ids)
            self.coefficients = list(coefficients)
            self.constant = constant
            self.operator = operator
            self.weight = weight

            self._rule_type = None

            # Currently only binary equality is accepted (and even that is a pretty approximate representation).
            if ((len(self.atom_ids) == 2) and (self.operator == '=')
                    and math.isclose(self.constant, 0.0) and math.isclose(self.coefficients[0], -self.coefficients[1])):
                self._rule_type = self.TYPE_BINARY_EQUALITY
                self.constant = round(self.constant)
            elif ((len(self.atom_ids) == 1) and (self.operator == '=')
                    and (math.isclose(self.constant, 0.0) or math.isclose(self.constant, 1.0))
                    and (math.isclose(self.coefficients[0], 1.0) or math.isclose(self.coefficients[0], -1.0))):
                self._rule_type = self.TYPE_FIXED_BINARY_VALUE

                self.constant = round(self.constant)
                self.coefficients[0] = round(self.coefficients[0])

                if (self.coefficients[0] == -1):
                    self.coefficients[0] = -self.coefficients[0]
                    self.constant = (self.constant + 1) % 2

            if (self._rule_type is None):
                raise NotImplementedError("Arithmetic Rules: [%s]" % (str(self)))

        # TODO(eriq): Weight is not included.
        def to_problog(self, atoms, queries, rng):
            rules = []

            if (self._rule_type == self.TYPE_BINARY_EQUALITY):
                # Write two rules, one with each atom in the head.
                for head_index in range(len(self.atom_ids)):
                    head_atom = atoms[self.atom_ids[head_index]]
                    body_atom = atoms[self.atom_ids[(head_index + 1) % len(self.atom_ids)]]

                    rules.append("1.0 :: %s :- %s ." % (head_atom.to_problog(), body_atom.to_problog()))
            elif (self._rule_type == self.TYPE_FIXED_BINARY_VALUE):
                prefix = ''
                if (self.coefficients[0] < 0):
                    prefix = '\\+'
                rules.append("1.0 :: %s%s ." % (prefix, atoms[self.atom_ids[0]].to_problog()))
            else:
                raise NotImplementedError("Unknown arithmetic rules type (%s): [%s]" % (self._rule_type, str(self)))

            return ' '.join(rules)

        def __repr__(self):
            operands = []
            for i in range(len(self.atom_ids)):
                operands.append("%f * <%d>" % (self.coefficients[i], self.atom_ids[i]))

            return "%s %s %f" % (' + '.join(operands), self.operator, self.constant)

    def _make_rule(self, ground_info):
        weight = ground_info['weight']
        if (weight < 0.0):
            weight = BaseGroundProbLog.HARD_WEIGHT

        if (ground_info['operator'] == '|'):
            return BaseGroundProbLog._LogicalRule(ground_info['atoms'], ground_info['coefficients'], weight)

        return BaseGroundProbLog._ArithmeticRule(ground_info['atoms'], ground_info['coefficients'], ground_info['constant'], ground_info['operator'], weight)

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
            return "%s(%s)" % (self.relation.name().lower(), ','.join(map(lambda x: '"' + str(x).lower().replace('"', '\\"') + '"', self.arguments)))

        def __repr__(self):
            operator = '==' if self.observed else '?='
            return "%s(%s) %s %s" % (self.relation.name(), ', '.join(map(str, self.arguments)), operator, self.value)
