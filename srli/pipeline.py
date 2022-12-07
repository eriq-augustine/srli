import json
import os
import re
import sys

import sklearn.metrics

import srli.engine.psl.engine
import srli.relation
import srli.util

class Pipeline(object):
    """
    A container that describes how to define, run, and evaluate a SRLi model.
    Currently, the capabilities of describing, building, and running a pipeline are very limited (as a prototype).
    """

    def __init__(self, options, rules, weights, squared, relations, learn_data, infer_data, evaluations):
        self._options = options
        self._rules = rules
        self._weights = weights
        self._squared = squared
        self._relations = relations
        self._learn_data = learn_data
        self._infer_data = infer_data
        self._evaluations = evaluations

    def run(self, engine_type = srli.engine.psl.engine.PSL):
        engine = engine_type(self._relations, self._rules, weights = self._weights, squared = self._squared)

        if ((len(self._learn_data) > 0) and (self._learn_data != self._infer_data)):
            self._learn(engine)

        if (len(self._infer_data) > 0):
            self._infer(engine)
            pass

    def _learn(self, engine):
        for relation in self._relations:
            relation.clear_data()

        for (relation, data) in self._learn_data.items():
            for (data_type, paths) in data.items():
                for path in paths:
                    relation.add_data_file(path, data_type = data_type)

        engine.learn(additional_config = self._options)

    def _infer(self, engine):
        for relation in self._relations:
            relation.clear_data()

        for (relation, data) in self._infer_data.items():
            for (data_type, paths) in data.items():
                for path in paths:
                    relation.add_data_file(path, data_type = data_type)

        results = engine.solve(additional_config = self._options)

        self._eval(results)

    def _eval(self, results):
        for (relation, info) in self._evaluations:
            if (info == 'CategoricalEvaluator'):
                expected, predicted, _ = srli.util.get_eval_categories(relation, results[relation])
                accuracy = sklearn.metrics.accuracy_score(expected, predicted)
                print('Categorical accuracy for %s: %f' % (relation, accuracy))
            elif (info == 'DiscreteEvaluator'):
                expected, predicted = srli.util.get_eval_values(relation, results[relation], discretize = True)
                f1 = sklearn.metrics.f1_score(expected, predicted)
                print('F1 for %s: %f' % (relation, f1))
            else:
                raise ValueError("Unknown evaluation: '%s'." % (info))

    # TODO(eriq): This assumes the config is syntactically/semantically correct, only minimal error checking is done.
    @staticmethod
    def from_psl_config(path):
        contents = []
        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if (line == '' or line.startswith('#') or line.startswith('//')):
                    continue

                if ('/*' in line):
                    raise ValueError("Multi-line comments ('/* ... */') not allowed.")

                contents.append(line)

        config = json.loads(' '.join(contents))

        base_path = os.path.dirname(path)

        options = Pipeline._parse_options(config)
        relations, learn_data, infer_data, evaluations = Pipeline._parse_relations(config, base_path)
        rules, weights, squared = Pipeline._parse_rules(config, relations)

        return Pipeline(options, rules, weights, squared, relations, learn_data, infer_data, evaluations)

    @staticmethod
    def _parse_options(config):
        if ('options' not in config):
            return {}

        options = config['options']

        if (('learn' in config) and ('options' in config['learn']) and len(config['learn']['options']) > 0):
            raise ValueError("Options can only be defined in the top-level options, found 'learn.options'.")

        if (('infer' in config) and ('options' in config['infer']) and len(config['infer']['options']) > 0):
            raise ValueError("Options can only be defined in the top-level options, found 'infer.options'.")

        return options

    @staticmethod
    def _parse_rules(config, relations):
        base_rules = config['rules']

        if (('learn' in config) and ('rules' in config['learn']) and len(config['learn']['rules']) > 0):
            raise ValueError("Rules can only be defined in the top-level rules, found 'learn.rules'.")

        if (('infer' in config) and ('rules' in config['infer']) and len(config['infer']['rules']) > 0):
            raise ValueError("Rules can only be defined in the top-level rules, found 'infer.rules'.")

        rules = []
        weights = []
        squared = []

        relation_map = {relation.name().upper() : relation for relation in relations}

        for base_rule in base_rules:
            base_rule = re.sub(r'\s+', ' ', base_rule).strip()

            match = re.search(r'^(\d+(?:\.\d+)?)\s*:\s*(.+?)\s*((?:\^[12])?)$', base_rule)
            if (match is not None):
                base_rule = match.group(2).strip()
                weight = float(match.group(1))
                modifier = match.group(3)

                # Check for a prior
                match = re.search(r'^!(\w+)\([^\)]+\)$', base_rule)
                if (match is not None):
                    name = match.group(1).upper()
                    relation_map[name].set_negative_prior_weight(weight)
                    continue

                rules.append(base_rule)
                weights.append(weight)

                if (modifier == '^2'):
                    squared.append(True)
                else:
                    squared.append(False)

                continue

            match = re.search(r'^(.+?)\s*\.$', base_rule)
            if (match is not None):
                base_rule = match.group(1).strip()

                # Check for a functional constraint.
                match = re.search(r'^(\w+)\([^\)]+\)\s*=\s*1(\.0)?$', base_rule)
                if (match is not None):
                    name = match.group(1).upper()
                    relation_map[name].set_functional(True)
                    continue

                rules.append(base_rule)
                weights.append(None)
                squared.append(None)

                continue

            raise ValueError("Could not parse rule: [%s]." % (base_rule))

        return rules, weights, squared

    @staticmethod
    def _parse_relations(config, base_path):
        """
        Returns:
            [relation, ...]
            {relation: {Relation.DataType: [path, ...], ...}, ...}
            {relation: {Relation.DataType: [path, ...], ...}, ...}
            [(relation, eval?), ...]
        """

        relations = []
        all_learn_data = {}
        all_infer_data = {}
        all_evaluations = []

        for (name, relation_config) in config['predicates'].items():
            relation, learn_data, infer_data, evaluations = Pipeline._parse_relation(name, relation_config, base_path)

            relations.append(relation)
            all_learn_data[relation] = learn_data
            all_infer_data[relation] = infer_data
            all_evaluations += evaluations

        return relations, all_learn_data, all_infer_data, all_evaluations

    # TODO(eriq): Types are ignored, arity consistency is not checked.
    @staticmethod
    def _parse_relation(name, relation_config, base_path):
        arity = -1

        if ('/' in name):
            parts = name.split('/')

            name = parts[0]
            arity = int(parts[1])

        if (('types' in relation_config) and (relation_config['types']) > 0):
            arity = len(relation_config['types'])

        relation = srli.relation.Relation(name, arity = arity)
        learn_data = {}
        infer_data = {}
        evaluations = []

        data_types = [
            ('observations', srli.relation.Relation.DataType.OBSERVED),
            ('targets', srli.relation.Relation.DataType.UNOBSERVED),
            ('truth', srli.relation.Relation.DataType.TRUTH),
        ]

        for (key, data_type) in data_types:
            if (key in relation_config):
                Pipeline._parse_data_spec(relation_config[key], data_type, learn_data, infer_data, base_path)

        if (('evaluations' in relation_config) and len(relation_config['evaluations']) > 0):
            evaluations += Pipeline._parse_evaluations(relation_config['evaluations'], relation)

        return relation, learn_data, infer_data, evaluations

    # TODO(eriq): Embeded data is not yet supported.
    @staticmethod
    def _parse_data_spec(data_config, data_type, learn_data, infer_data, base_path):
        # Data in the 'all' partition (or when no partition is specified) goes in both.

        if (isinstance(data_config, list)):
            data_config = {'all': data_config}

        for (partition, raw_paths) in data_config.items():
            paths = []
            for path in raw_paths:
                if (not isinstance(path, str)):
                    raise ValueError('Only paths currently allowed as data.')

                if (not os.path.isabs(path)):
                    path = os.path.join(base_path, path)

                paths.append(os.path.normpath(path))

            if (partition == 'learn'):
                Pipeline._add_data_spec(learn_data, data_type, paths)
            elif (partition == 'infer'):
                Pipeline._add_data_spec(infer_data, data_type, paths)
            elif (partition == 'all'):
                Pipeline._add_data_spec(learn_data, data_type, paths)
                Pipeline._add_data_spec(infer_data, data_type, paths)
            else:
                raise ValueError("Unknown data phase: '%s'." % (partition))

    @staticmethod
    def _add_data_spec(data, data_type, new_list):
        if (data_type not in data):
            data[data_type] = []

        data[data_type] += new_list

    # TODO(eriq): Options are ignored.
    @staticmethod
    def _parse_evaluations(eval_configs, relation):
        evaluations = []

        for eval_config in eval_configs:
            if (isinstance(eval_config, str)):
                evaluations.append((relation, eval_config))
            elif (isinstance(eval_config, dict)):
                evaluations.append((relation, eval_config['evaluator']))
            else:
                raise ValueError("Unknown eval config: '%s'."% (eval_config))

        return evaluations

def main(path):
    pipeline = Pipeline.from_psl_config(path)
    pipeline.run()

def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <config path>" % (executable), file = sys.stderr)
        sys.exit(1)

    return args.pop(0)

if (__name__ == '__main__'):
    main(_load_args(sys.argv))
