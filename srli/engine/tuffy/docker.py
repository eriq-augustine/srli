import os
import re
import shutil
import string
import tempfile

import docker

import srli.engine.base

EVIDENCE_FILENAME = 'evidence.db'
PROGRAM_FILENAME = 'prog.mln'
QUERY_FILENAME = 'query.db'
OUTPUT_FILENAME = 'out.txt'

TEMP_DIR_PREFIX = 'srli.tuffy.'

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
LIB_DIR = os.path.join(THIS_DIR, 'lib')

DOCKER_TAG = 'srli.tuffy'
DOCKER_TUFFY_IO_DIR = '/tuffy/io'

class Tuffy(srli.engine.base.BaseEngine):
    """
    Run Tuffy in a Docker container.
    """

    def __init__(self, relations, rules, weights = None, **kwargs):
        super().__init__(relations, rules, **kwargs)

        if (weights is not None and len(weights) > 0):
            self._weights = weights
        else:
            self._weights = [1.0] * len(self._rules)

    def solve(self, cleanup_files = True, **kwargs):
        temp_dir = tempfile.mkdtemp(prefix = TEMP_DIR_PREFIX)

        program_path = os.path.join(temp_dir, PROGRAM_FILENAME)
        evidence_path = os.path.join(temp_dir, EVIDENCE_FILENAME)
        query_path = os.path.join(temp_dir, QUERY_FILENAME)
        output_path = os.path.join(temp_dir, OUTPUT_FILENAME)

        self._write_program(program_path)
        self._write_evidence(evidence_path)
        self._write_query(query_path)

        self._run_tuffy(temp_dir)
        raw_results = self._read_results(output_path)

        if (cleanup_files):
            shutil.rmtree(temp_dir)

        results = {}
        for relation in self._relations:
            if (not relation.has_unobserved_data()):
                continue

            results[relation] = []

            for row in relation.get_unobserved_data():
                key = tuple(row[0:relation.arity()])
                if (key in raw_results[relation]):
                    results[relation].append(list(key) + [raw_results[relation][key]])
                else:
                    results[relation].append(list(key) + [0.0])

        return results

    def _write_file(self, path, lines):
        with open(path, 'w') as file:
            for line in lines:
                file.write(str(line) + "\n")

    def _find_relation(self, name):
        for relation in self._relations:
            if (relation.name().lower() == name.lower()):
                return relation
        return None

    def _read_results(self, path, has_value = False):
        results = {}

        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if (line == ''):
                    continue

                parts = line.split("\t")
                if (has_value):
                    atom = parts[0]
                    value = float(parts[1])
                else:
                    atom = parts[0]
                    value = 1.0

                (predicate, _, arguments) = atom.partition('(')
                relation = self._find_relation(predicate)
                arguments = tuple(arguments.rstrip(')').replace('"', '').split(', '))

                if (relation not in results):
                    results[relation] = {}

                results[relation][arguments] = value

        return results

    def _write_program(self, path):
        program = []
        has_prior = False

        for relation in self._relations:
            has_prior |= relation.has_negative_prior_weight()
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

        # Write any prior rules.
        if (has_prior):
            program.append('')
            for relation in self._relations:
                if (relation.has_negative_prior_weight()):
                    arguments = ', '.join([value for value in string.ascii_lowercase[0:relation.arity()]])
                    program.append("%f !%s(%s)" % (relation.get_negative_prior_weight(), relation.name(), arguments))

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

                line = "%s(%s)" % (relation.name(), ', '.join(map(str, row[0:relation.arity()])))
                query.append(line)

        self._write_file(path, query)

    # TODO(eriq): There are several alternate paths to using Docker (e.g. a prebuilt image).
    def _run_tuffy(self, io_dir):
        client = docker.from_env()

        # Build the image (Docker's cache will be used for subsequent runs).
        client.images.build(path = LIB_DIR, tag = DOCKER_TAG, rm = True, quiet = False)

        try:
            container = client.containers.get(DOCKER_TAG)
            container.remove()
        except docker.errors.NotFound as ex:
            pass

        # Run the container with the temp dir as a mount.
        volumes = {
            io_dir: {
                'bind': DOCKER_TUFFY_IO_DIR,
                'mode': 'rw',
            },
        }

        client.containers.run(DOCKER_TAG, volumes = volumes, name = DOCKER_TAG,
                remove = True, network_disabled = True)
