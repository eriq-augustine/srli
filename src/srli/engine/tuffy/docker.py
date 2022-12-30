import atexit
import copy
import math
import functools
import os
import re
import shutil
import string
import tempfile
import uuid

import docker

import srli.engine.base
import srli.parser

EVIDENCE_FILENAME = 'evidence.db'
PROGRAM_FILENAME = 'prog.mln'
QUERY_FILENAME = 'query.db'
OUTPUT_FILENAME = 'out.txt'

TEMP_DIR_PREFIX = 'srli.tuffy.'

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
LIB_DIR = os.path.join(THIS_DIR, 'lib')

DOCKER_TAG = 'srli.tuffy'
DOCKER_TUFFY_IO_DIR = '/tuffy/io'

# Tuffy variables must start with a lower case letter.
DUMMY_VARIABLE_PREFIX = 'var_srli__'
# Tuffy constants must be only integer or start with an upper case letter.
DUMMY_CONSTANT_PREFIX = 'CON_srli__'

# TODO(eriq): Partial functionals are ignored.
class Tuffy(srli.engine.base.BaseEngine):
    """
    Run Tuffy in a Docker container.
    """

    def __init__(self, relations, rules, cleanup_files = True, include_priors = True, **kwargs):
        super().__init__(relations, rules, **kwargs)

        self._cleanup_files = cleanup_files
        self._include_priors = include_priors

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
            self._check_output(output_path)
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
            self._check_output(output_path)
            raw_results = self._read_results(output_path)
        finally:
            self._cleanup(temp_dir)

        results = {}
        for relation in self._relations:
            if (not relation.has_unobserved_data()):
                continue

            values = []

            for row in relation.get_unobserved_data():
                key = self._convert_source_atom(relation, row)

                if (key in raw_results):
                    value = raw_results[key]
                else:
                    # Tuffy does not output zeros.
                    value = 0.0

                values.append(list(row) + [value])

            results[relation] = values

        return results

    def _check_output(self, output_path):
        if (not os.path.isfile(output_path)):
            raise RuntimeError("Tuffy did not complete successfully.")

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

    def _read_results(self, path):
        raw_results = {}

        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if (line == ''):
                    continue

                parts = line.split("\t")
                atom = parts[0]
                value = 1.0

                raw_results[atom] = value

        return raw_results

    def _write_program(self, path):
        program = []
        has_prior = False

        for relation in self._relations:
            has_prior |= relation.has_negative_prior_weight()

            arguments = [str(variable_type).lower() for variable_type in relation.variable_types()]

            # Check for a hard summation constraint.
            if (relation.has_sum_constraint() and relation.sum_constraint().is_hard_functional()):
                for index in relation.sum_constraint().label_indexes:
                    arguments[index] += '!'

            predicate = "%s(%s)" % (relation.name().upper(), ', '.join(arguments))

            if (relation.is_observed()):
                predicate = '*' + predicate
            program.append(predicate)

        program.append('')

        for rule in self._rules:
            rule_texts = self._convert_rule(rule)
            if (rule_texts is None):
                continue

            for rule_text in rule_texts:
                if (rule.is_weighted()):
                    program.append("%f %s" % (rule.weight(), rule_text))
                else:
                    program.append("%s ." % (rule_text))

        # Write any prior rules.
        if (self._include_priors and has_prior):
            program.append('')
            for relation in self._relations:
                if (relation.has_negative_prior_weight()):
                    arguments = ', '.join([value for value in string.ascii_lowercase[0:relation.arity()]])
                    program.append("%f !%s(%s)" % (relation.get_negative_prior_weight(), relation.name().upper(), arguments))

        self._write_file(path, program)

    def _convert_rule(self, rule):
        parsed_rule = srli.parser.parse(rule.text())

        if (isinstance(parsed_rule, srli.parser.DNF)):
            atoms = [self._convert_parser_atom(atom) for atom in parsed_rule.atoms]
            return [' v '.join(atoms)]

        if (not isinstance(parsed_rule, srli.parser.LinearRelation)):
            raise RuntimeError("Unknown rule type: '%s' (%s)." % (rule, type(parsed_rule)))

        # Only specific types of arithmetic rules are allowed.

        # Binary equality.
        if ((parsed_rule.operator == '=') and (math.isclose(parsed_rule.constant, 0.0)) and
                (len(parsed_rule.atoms) == 2) and (math.isclose(parsed_rule.atoms[0].modifier, -parsed_rule.atoms[1].modifier))):
            atoms = []
            for parser_atom in parsed_rule.atoms:
                atom = copy.copy(parser_atom)
                atom.modifier = 1
                atoms.append(self._convert_parser_atom(atom))

            return [
                "%s => %s" % (atoms[0], atoms[1]),
                "%s => %s" % (atoms[1], atoms[0]),
            ]

        # Fixed binary value.
        if ((parsed_rule.operator == '=') and (math.isclose(parsed_rule.constant, 0.0) or math.isclose(parsed_rule.constant, 1.0)) and
                (len(parsed_rule.atoms) == 1)):
            constant = 0 if (math.isclose(parsed_rule.constant, 0.0)) else 1
            atom = copy.copy(parsed_rule.atoms[0])

            if (atom.modifier < 0):
                atom.modifier = -atom.modifier
                constant = (constant + 1) % 2

            prefix = ''
            if (atom.modifier < 0):
                prefix = '!'

            return ["%s%s" % (prefix, self._convert_parser_atom(atom))]

        raise RuntimeError("This form of linear rule is not supported in Tuffy: '%s'." % (rule))

    # Convert an atom that comes from srli.parser.
    def _convert_parser_atom(self, parser_atom):
        relation_name = parser_atom.relation_name.upper()

        modifier = ''
        if (parser_atom.modifier < 0):
            modifier = '!'

        arguments = []
        for raw_arg in parser_atom.arguments:
            if (isinstance(raw_arg, srli.parser.Variable)):
                arguments.append(self._convert_variable(str(raw_arg)))
            elif (isinstance(raw_arg, srli.parser.Constant)):
                arguments.append(self._convert_constant(str(raw_arg)))
            else:
                raise RuntimeError("Unknown atom argument: '%s' (%s)." % (raw_arg, type(raw_arg)))

        return "%s%s(%s)" % (modifier, relation_name, ', '.join(arguments))

    # Convert an atom that comes from a relations's data (i.e. a row).
    def _convert_source_atom(self, relation, source_atom):
        relation_name = relation.name().upper()
        arguments = [self._convert_constant(str(arg)) for arg in source_atom[0:relation.arity()]]

        return "%s(%s)" % (relation_name, ', '.join(arguments))

    def _convert_constant(self, text):
        text = text.replace('"', '\\"')
        return '"' + DUMMY_CONSTANT_PREFIX + text + '"'

    def _convert_variable(self, text):
        return DUMMY_VARIABLE_PREFIX + text

    def _write_evidence(self, path):
        evidence = []

        for relation in self._relations:
            if (not relation.has_observed_data()):
                continue

            for row in relation.get_observed_data():
                args = list(map(lambda argument: self._convert_constant(argument), row[0:relation.arity()]))
                line = "%s(%s)" % (relation.name().upper(), ', '.join(map(str, args)))

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
                args = list(map(lambda argument: self._convert_constant(argument), row[0:relation.arity()]))
                line = "%s(%s)" % (relation.name().upper(), ', '.join(map(str, args)))
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

            stop_container_partial = functools.partial(Tuffy._stop_container, container_id)
            atexit.register(stop_container_partial)

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
            Tuffy._stop_container(container_id)

    @staticmethod
    def _stop_container(container_id):
        client = docker.from_env()

        try:
            container = client.containers.get(container_id)
        except docker.errors.NotFound:
            return

        if (container.status == 'running'):
            container.stop()
