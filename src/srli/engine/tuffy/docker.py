import os
import re
import shutil
import string
import tempfile
import uuid

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

    def __init__(self, relations, rules, cleanup_files = True, **kwargs):
        super().__init__(relations, rules, **kwargs)

        self._cleanup_files = cleanup_files

        missing_types = False
        for relation in self._relations:
            if (relation.variable_types() is None):
                missing_types = True

        if (missing_types):
            print("Warning: Required types are missing for Tuffy, inferring types.")
            self._infer_variable_types()

    def learn(self, max_iterations = None, **kwargs):
        temp_dir, output_path = self._prep_run()

        args = ['-learnwt']

        if (max_iterations is not None):
            args += ['-dMaxIter', str(max_iterations)]

        try:
            self._run_tuffy(temp_dir, additional_args = args)
            weights = self._parse_weights(output_path)
        finally:
            self._cleanup(temp_dir)

        for i in range(len(self._rules)):
            self._rules[i].set_weight(weights[i])

        return self

    def solve(self, **kwargs):
        temp_dir, output_path = self._prep_run()

        try:
            self._run_tuffy(temp_dir)
            raw_results = self._read_results(output_path)
        finally:
            self._cleanup(temp_dir)

        results = {}
        for relation in self._relations:
            if (not relation.has_unobserved_data()):
                continue

            results[relation] = []

            for row in relation.get_unobserved_data():
                key = tuple(row[0:relation.arity()])

                # The relation may not be in the results if Tuffy outputs no true values,
                # and the key may not be in the results if its specific value is false.
                if ((relation not in raw_results) or (key not in raw_results[relation])):
                    results[relation].append(list(key) + [0.0])
                else:
                    results[relation].append(list(key) + [raw_results[relation][key]])

        return results

    def _prep_run(self):
        temp_dir = tempfile.mkdtemp(prefix = TEMP_DIR_PREFIX)

        program_path = os.path.join(temp_dir, PROGRAM_FILENAME)
        evidence_path = os.path.join(temp_dir, EVIDENCE_FILENAME)
        query_path = os.path.join(temp_dir, QUERY_FILENAME)
        output_path = os.path.join(temp_dir, OUTPUT_FILENAME)

        self._write_program(program_path)
        self._write_evidence(evidence_path)
        self._write_query(query_path)

        return temp_dir, output_path

    def _cleanup(self, temp_dir):
        if (self._cleanup_files):
            shutil.rmtree(temp_dir)

    def _write_file(self, path, lines):
        with open(path, 'w') as file:
            for line in lines:
                file.write(str(line) + "\n")

    def _find_relation(self, name):
        for relation in self._relations:
            if (relation.name().upper() == name.upper()):
                return relation
        return None

    def _parse_weights(self, path):
        ordered_weights = []

        relation_map = {relation.name().upper() : relation for relation in self._relations}

        with open(path, 'r') as file:
            skip = True

            for line in file:
                if ('WEIGHT OF LAST ITERATION' in line):
                    skip = False
                    continue
                elif (skip):
                    continue

                line = line.strip()
                if (line == ''):
                    continue

                # Check for priors first.
                match = re.search(r'^(-?\d+(?:\.\d+))\s+!(\w+)\([^\)]+\)\s+\/\/(\d+\.\d+)$', line)
                if (match is not None):
                    weight = float(match.group(1))
                    relation_name = match.group(2).upper()

                    if (relation_name not in relation_map):
                        raise ValueError("Could not find relation (%s) found in prior: '%s'." % (relation_name, line))

                    relation_map[relation_name].set_negative_prior_weight(weight)

                    continue

                # Soft rules.
                match = re.search(r'^(-?\d+(?:\.\d+))\s+.+?\s+\/\/(\d+\.\d+)$', line)
                if (match is not None):
                    weight = float(match.group(1))
                    index = float(match.group(2))

                    ordered_weights.append((index, weight))

                    continue

                # Hard rules.
                match = re.search(r' \. \/\/(\d+\.\d+)hardfixed$', line)
                if (match is not None):
                    index = float(match.group(1))

                    ordered_weights.append((index, None))

                    continue

                raise ValueError("Could not parse learned Tuffy weight from output rule: '%s'." % (line))

        # Sort the weights according to the index output by Tuffy, which should match the order they were inserted.
        return [weight for (index, weight) in sorted(ordered_weights)]

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

            arguments = list(map(str, relation.variable_types()))

            # Check for a hard summation constraint.
            if (relation.has_sum_constraint() and relation.sum_constraint().is_hard_functional()):
                for index in relation.sum_constraint().label_indexes:
                    arguments[index] += '!'

            predicate = "%s(%s)" % (relation.name().upper(), ', '.join(arguments))

            if (relation.is_observed()):
                predicate = '*' + predicate
            program.append(predicate)

        program.append('')

        for i in range(len(self._rules)):
            rule = self._rules[i].text()
            rule = rule.replace('&', ',')
            rule = rule.replace('->', '=>')
            rule = rule.replace('>>', '=>')
            rule = rule.replace(' = ', ' => ')
            # Constants can use double quotes.
            rule = rule.replace('\'', '"')
            rule = re.sub(r',\s*\(\w+\s*!=\s*\w+\)', '', rule)

            # TODO(eriq): Rule variables must be all lower case.
            # HACK(eriq): This method is very quick and dirty.
            rule = rule.lower()

            for relation in self._relations:
                rule = re.sub(r'\b' + relation.name().lower() + r'\b', relation.name().upper(), rule)

            if (self._rules[i].is_weighted()):
                program.append("%f %s" % (self._rules[i].weight(), rule))
            else:
                program.append("%s ." % (rule))

        # Write any prior rules.
        if (has_prior):
            program.append('')
            for relation in self._relations:
                if (relation.has_negative_prior_weight()):
                    arguments = ', '.join([value for value in string.ascii_lowercase[0:relation.arity()]])
                    program.append("%f !%s(%s)" % (relation.get_negative_prior_weight(), relation.name().upper(), arguments))

        self._write_file(path, program)

    def _write_evidence(self, path):
        evidence = []

        for relation in self._relations:
            if (not relation.has_observed_data()):
                continue

            for row in relation.get_observed_data():
                # Tuffy args cannot have spaces.
                row = list(map(lambda argument: argument.replace(' ', '_'), row))

                line = "%s(%s)" % (relation.name().upper(), ', '.join(map(str, row[0:relation.arity()])))

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

                line = "%s(%s)" % (relation.name().upper(), ', '.join(map(str, row[0:relation.arity()])))
                query.append(line)

        self._write_file(path, query)

    def _run_tuffy(self, io_dir, additional_args = []):
        client = docker.from_env()

        # Build the image (Docker's cache will be used for subsequent runs).
        client.images.build(path = LIB_DIR, tag = DOCKER_TAG, rm = True, quiet = False)

        # Run the container with the temp dir as a mount.
        volumes = {
            io_dir: {
                'bind': DOCKER_TUFFY_IO_DIR,
                'mode': 'rw',
            },
        }

        container_id = DOCKER_TAG + '_' + str(uuid.uuid4())
        container = None

        try:
            # Ideally we would disable all networking (network_disabled = True),
            # but Tuffy will throw an error.
            container = client.containers.run(DOCKER_TAG, command = additional_args, volumes = volumes, name = container_id,
                    remove = True, network_disabled = False,
                    detach = True)

            for line in container.logs(stream = True):
                print(line.decode(), end = '')
            print()
        except Exception as ex:
            if (container is not None):
                print('Tuffy container failed to run, dumping remaining log.')
                logs = container.logs()
                if (logs is not None):
                    for line in logs:
                        print(line.decode(), end = '')
                    print()

            raise ex
        finally:
            if (container is not None):
                container.wait()
                container = None
