import string

import problog

import srli.parser

# HACK(eriq): All relations and arguments are made lowercase. There is potneital for collision in case-sensitive data.
# TODO(eriq): There is still a ton of work on when to introduce layers. As is, many programs (e.g. transitive ones) will finish.

DUMMY_VALUE_SUFFIX = '__dummy__'

class ProbLog(object):
    """
    A basic implementation that builds a problog model and calls into the Python interface.
    """

    def __init__(self, relations, rules, weights = None, **kwargs):
        self._relations = relations
        self._rules = rules

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [None] * len(self._rules)

    def solve(self, **kwargs):
        program = []

        self._create_rules(program)
        self._add_observed_data(program)
        self._add_unobserved_data(program)
        results = self._eval("\n".join(program))

        # print("\n".join(program))

        return results

    def _eval(self, text):
        program = problog.program.PrologString(text)

        raw_results = problog.get_evaluatable().create_from(program).evaluate()
        raw_results = {str(key): float(value) for (key, value) in raw_results.items()}

        results = {}
        for relation in self._relations:
            if (not relation.has_unobserved_data()):
                continue

            results[relation] = []

            data = relation.get_unobserved_data()
            pl_atoms, _ = self._create_atoms(relation, data)

            for i in range(len(data)):
                row = data[i]
                pl_atom = pl_atoms[i]

                result = raw_results[pl_atoms[i]]
                atom = data[i][0:relation.arity()] + [result]
                results[relation].append(atom)

        return results

    def _add_observed_data(self, program):
        program.append("\n% Observed Data")

        for relation in self._relations:
            if (relation.has_observed_data()):
                data = relation.get_observed_data()
            elif (relation.has_unobserved_data()):
                # Fully unobsverd relations need a dummy value to allow ProbLog to work with shells.
                # This is a workaround for ProbLog errors.
                data = [[value + DUMMY_VALUE_SUFFIX for value in string.ascii_lowercase[0:relation.arity()]]]
            else:
                continue

            program.append('')

            level = None
            if (relation.has_unobserved_data()):
                level = 1

            atoms, values = self._create_atoms(relation, data, level = level)
            for i in range(len(atoms)):
                program.append("%f :: %s ." % (values[i], atoms[i]))

    def _add_unobserved_data(self, program):
        program.append("\n% Unobserved Data")

        for relation in self._relations:
            if (not relation.has_unobserved_data()):
                continue

            program.append('')

            atoms, _ = self._create_atoms(relation, relation.get_unobserved_data())
            for i in range(len(atoms)):
                program.append("query(%s) ." % (atoms[i]))

    def _create_atoms(self, relation, data, level = None):
        atoms = []
        values = []

        suffix = ''
        if (level is not None):
            suffix = "_l%d" % (level)

        for row in data:
            value = 1.0
            if (len(row) > relation.arity()):
                value = float(row[-1])

            arguments = ', '.join([arg.lower() for arg in map(str, row[0:relation.arity()])])

            atoms.append("%s%s(%s)" % (relation.name().lower(), suffix, arguments))
            values.append(value)

        return atoms, values

    def _create_rules(self, program):
        unobserved_relation_names = set()
        for relation in self._relations:
            if (relation.has_unobserved_data()):
                unobserved_relation_names.add(relation.name())

        # [(number of unobserved atoms, rule index), ...]
        collectivity_counts = []
        # [rule index, ...]
        non_collective_rules = []

        for i in range(len(self._rules)):
            ast = srli.parser.parse(self._rules[i])

            unobserved_atoms = 0
            for atom in ast.get_atoms():
                if (atom['relation_name'] in unobserved_relation_names):
                    unobserved_atoms += 1

            if (unobserved_atoms <= 1):
                non_collective_rules.append(i)
            else:
                collectivity_counts.append((unobserved_atoms, i))

        # Handle lower collectivity rules first.
        collectivity_counts.sort()

        # Bring in all rules at the same layer and then use priors to transition to the final layer.

        for rule_index in non_collective_rules:
            program.append("% Non-collective rule: " + self._rules[rule_index])
            program.append(self._simple_rule_rename(rule_index, unobserved_relation_names, 1))

        for (_, rule_index) in collectivity_counts:
            if (len(program) > 0):
                program.append('')

            program.append("% Collective rule: " + self._rules[rule_index])
            program += self._collective_rule_rename(rule_index, unobserved_relation_names, 1, 1)

        # Add in the transition from the last layer to the actual query as well as priors.

        program.append('\n% Final Layer Transitions and Priors')

        relation_map = {relation.name(): relation for relation in self._relations}
        for relation_name in sorted(unobserved_relation_names):
            relation = relation_map[relation_name]
            arguments = ', '.join(string.ascii_uppercase[0:relation.arity()])

            program.append('')
            program.append("%s(%s) :- %s_l%d(%s) ." % (relation_name.lower(), arguments, relation_name.lower(), 1, arguments))

            if (relation.has_negative_prior_weight()):
                program.append("%f :: %s(%s) :- \\+ %s_l%d(%s) ." % (relation.get_negative_prior_weight(),
                        relation_name.lower(), arguments, relation_name.lower(), 1, arguments))

    def _collective_rule_rename(self, rule_index, unobserved_relation_names, current_level, next_level):
        rule_strings = []
        ast = srli.parser.parse(self._rules[rule_index])

        if (not isinstance(ast, srli.parser.Implication)):
            raise ValueError("Expected rule to be implication, got: '%s'." % (self._rules[rule_index]))

        head_relations = set([atom['relation_name'] for atom in ast[1].get_atoms()])
        body_relations = set([atom['relation_name'] for atom in ast[0].get_atoms()])

        rule_string = self._walk_rule(ast, unobserved_relation_names, current_level, next_level)

        weight = ''
        if (self._weights[rule_index] is not None):
            weight = "%f :: " % (self._weights[rule_index])

        rule_strings.append("%s%s ." % (weight, rule_string))

        if (current_level == next_level):
            return rule_strings

        # When moving levels, every unobserved relation needs to be moved through the current level.
        # For relations with the same relation in the head and body, the transition is already made.
        # For all other relations, the transition needs to be made explicitly.

        relation_map = {relation.name(): relation for relation in self._relations}

        for relation_name in unobserved_relation_names:
            relation = relation_map[relation_name]
            arguments = ', '.join(string.ascii_uppercase[0:relation.arity()])
            rule_strings.append("%s_l%d(%s) :- %s_l%d(%s) ." % (relation_name.lower(), next_level, arguments, relation_name.lower(), current_level, arguments))

        return rule_strings

    def _simple_rule_rename(self, rule_index, unobserved_relation_names, current_level):
        ast = srli.parser.parse(self._rules[rule_index])
        rule_string = self._walk_rule(ast, unobserved_relation_names, current_level, current_level)

        weight = ''
        if (self._weights[rule_index] is not None):
            weight = "%f :: " % (self._weights[rule_index])

        return "%s%s ." % (weight, rule_string)

    def _walk_rule(self, ast_node, unobserved_relation_names, current_level, next_level):
        if (isinstance(ast_node, srli.parser.Implication)):
            body = self._walk_rule(ast_node[0], unobserved_relation_names, current_level, None)
            head = self._walk_rule(ast_node[1], unobserved_relation_names, next_level, None)
            return "%s :- %s" % (head, body)
        elif (isinstance(ast_node, srli.parser.Conjunction)):
            operands = [self._walk_rule(child, unobserved_relation_names, current_level, next_level) for child in ast_node]
            return ', '.join(operands)
        elif (isinstance(ast_node, srli.parser.Disjunction)):
            operands = [self._walk_rule(child, unobserved_relation_names, current_level, next_level) for child in ast_node]
            return ' | '.join(operands)
        elif (isinstance(ast_node, srli.parser.Atom)):
            relation_name = ast_node['relation_name']
            if (relation_name in unobserved_relation_names):
                relation_name = "%s_l%d" % (relation_name, current_level)

            negation = ''
            if (ast_node['negated']):
                negation += '!'

            return "%s%s(%s)" % (negation, relation_name.lower(), ', '.join(ast_node['arguments']))
        else:
            raise ValueError("Unknown ast node type: %s." % (str(type(ast_node))))
