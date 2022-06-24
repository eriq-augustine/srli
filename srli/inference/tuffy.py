import os
import re
import subprocess
import string
import tempfile

EVIDENCE_FILENAME = 'evidence.db'
PROGRAM_FILENAME = 'prog.mln'
QUERY_FILENAME = 'query.db'
OUTPUT_FILENAME = 'out.txt'

# HACK(eriq): This is meant to be replaced with a native implementation.
TUFFY_JAR_PATH = os.path.join(os.getenv('TUFFY_HOME', 'tuffy'), 'binary', 'tuffy.jar')
TUFFY_CONFIG_PATH = os.path.join(os.getenv('TUFFY_HOME', 'tuffy'), 'tuffy.conf')

class Tuffy(object):
    def __init__(self, relations, rules, weights = None, **kwargs):
        self._relations = relations
        self._rules = rules

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [1.0] * len(self._rules)

    def _write_file(self, path, lines):
        with open(path, 'w') as file:
            for line in lines:
                file.write(str(line) + "\n")

    def _find_relation(self, name):
        for relation in self._relations:
            if (relation.name().lower() == name.lower()):
                return relation
        return None

    def _read_file(self, path):
        rows = []

        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if (line == ''):
                    continue

                (value, atom) = line.split("\t")
                value = float(value)

                (predicate, _, arguments) = atom.partition('(')
                relation = self._find_relation(predicate)
                arguments = arguments.rstrip(')').strip('"').split('", "')

                rows.append([relation] + arguments + [value])

        return rows

    def _write_program(self, path):
        program = []

        for relation in self._relations:
            predicate = "%s(%s)" % (relation.name(), ', '.join(map(str, relation.variable_types())))

            if (relation.is_observed()):
                predicate = '*' + predicate
            program.append(predicate)

        program.append('')

        for i in range(len(self._rules)):
            rule = self._rules[i]
            rule = rule.replace('&', ',')
            rule = rule.replace('->', '=>')
            rule = rule.replace(' = ', ' => ')
            rule = re.sub(r',\s*\(\w+\s*!=\s*\w+\)', '', rule)

            # TODO(eriq): Rule variables must be all lower case.
            # HACK(eriq): This method is very quick and dirty.
            rule = rule.lower()
            for relation in self._relations:
                rule = rule.replace(relation.name().lower(), relation.name())

            if (self._weights[i] is None):
                program.append("%s ." % (rule))
            else:
                program.append("%f %s" % (self._weights[i], rule))

        self._write_file(path, program)

    def _write_evidence(self, path):
        evidence = []

        for relation in self._relations:
            if (not relation.has_observed_data()):
                continue

            for row in relation.get_observed_data():
                # Tuffy args cannot have spaces.
                row = list(map(lambda argument: argument.replace(' ', '_'), row))

                line = "%s(%s)" % (relation.name(), ', '.join(map(str, row[0:relation.arity()])))

                if (len(row) > relation.arity()):
                    line = "%f %s" % (float(row[-1]), line)

                evidence.append(line)

        self._write_file(path, evidence)

    def _write_query(self, path):
        query = []

        for relation in self._relations:
            if (not relation.has_unobserved_data()):
                continue

            for row in relation.get_unobserved_data():
                # Tuffy args cannot have spaces.
                row = list(map(lambda argument: argument.replace(' ', '_'), row))

                prior = ''
                if (relation.has_negative_prior_weight()):
                    prior = "%f " % (relation.get_negative_prior_weight())

                line = "%s%s(%s)" % (prior, relation.name(), ', '.join(map(str, row[0:relation.arity()])))
                query.append(line)

        self._write_file(path, query)

    def _run_tuffy(self, program_path, evidence_path, query_path, output_path):
        command = "java -jar '%s' -conf '%s' -marginal -mln '%s' -evidence '%s' -queryFile '%s' -result '%s'" % (
                TUFFY_JAR_PATH, TUFFY_CONFIG_PATH,
                program_path, evidence_path, query_path, output_path)

        subprocess.run(command, shell = True)

    def solve(self, additional_config = None, **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            program_path = os.path.join(temp_dir, PROGRAM_FILENAME)
            evidence_path = os.path.join(temp_dir, EVIDENCE_FILENAME)
            query_path = os.path.join(temp_dir, QUERY_FILENAME)
            output_path = os.path.join(temp_dir, OUTPUT_FILENAME)

            self._write_program(program_path)
            self._write_evidence(evidence_path)
            self._write_query(query_path)

            self._run_tuffy(program_path, evidence_path, query_path, output_path)

            output = self._read_file(output_path)

        results = {}

        for row in output:
            predicate = row[0]

            if (predicate not in results):
                results[predicate] = []

            results[predicate].append(row[1:])

        return results
