import argparse
import json
import os
import re
import time

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

    def run(self, engine_type, additional_options = {}):
        options = dict(self._options)
        options.update(additional_options)

        print('!!!')
        print(self._evaluations)
        print('!!!')

        engine = engine_type(self._relations, self._rules,
                weights = self._weights, squared = self._squared,
                options = options, evaluations = self._evaluations)

        if ((len(self._learn_data) > 0) and (self._learn_data != self._infer_data)):
            self._learn(engine)

        if (len(self._infer_data) > 0):
            self._infer(engine)
            pass

    def __repr__(self):
        return json.dumps({
            'options': self._options,
            'rules': self._rules,
            'weights': self._weights,
            'squared': self._squared,
            'relations': [relation.to_dict() for relation in self._relations],
            'learn_data': {str(relation) : {str(data_type) : paths for (data_type, paths) in data_spec.items()} for (relation, data_spec) in self._learn_data.items()},
            'infer_data': {str(relation) : {str(data_type) : paths for (data_type, paths) in data_spec.items()} for (relation, data_spec) in self._infer_data.items()},
            'evaluations': [evaluation.to_dict() for evaluation in self._evaluations],
        }, indent = 4)

    def _learn(self, engine):
        print("%d -- Loading learning data." % (int(time.time())))

        for relation in self._relations:
            relation.clear_data()

        for (relation, data) in self._learn_data.items():
            for (data_type, paths) in data.items():
                for path in paths:
                    relation.add_data_file(path, data_type = data_type)

        print("%d -- Starting learning engine." % (int(time.time())))

        engine.learn()

    def _infer(self, engine):
        print("%d -- Loading inference data." % (int(time.time())))

        for relation in self._relations:
            relation.clear_data()

        for (relation, data) in self._infer_data.items():
            for (data_type, paths) in data.items():
                for path in paths:
                    relation.add_data_file(path, data_type = data_type)

        print("%d -- Starting inference engine." % (int(time.time())))

        results = engine.solve()

        self._eval(results)

    def _eval(self, results):
        print("%d -- Starting evaluation." % (int(time.time())))

        stats = []

        for evaluation in self._evaluations:
            stats.append((evaluation.metric_name(), evaluation.relation(), evaluation.evaluate(results)))

        for (metric, relation, value) in stats:
            print('Evaluation Result -- Metric: %s, Relation: %s, Value: %f' % (metric, relation.name(), value))

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
                match = re.search(r'^[!~](\w+)\([^\)]+\)$', base_rule)
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
                match = re.search(r'^(\w+)\([^\)]+\+[^\)]*\)\s*=\s*1(\.0)?$', base_rule)
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

        if (('types' in relation_config) and len(relation_config['types']) > 0):
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

    @staticmethod
    def _parse_evaluations(eval_configs, relation):
        eval_map = {value : key for (key, value) in srli.engine.psl.engine.PSL.EVAL_MAP.items()}

        evaluations = []

        for eval_config in eval_configs:
            if (isinstance(eval_config, str)):
                name = eval_config
                primary = False
                options = {}
            elif (isinstance(eval_config, dict)):
                name = eval_config['evaluator']

                primary = False
                if ('primary' in eval_config):
                    primary = bool(eval_config['primary'])

                options = {}
                if ('options' in eval_config):
                    options = eval_config['options']
            else:
                raise ValueError("Unknown eval config: '%s'."% (eval_config))

            if (name not in eval_map):
                raise ValueError("Unknown eval type: '%s'." % (name))

            evaluations.append(eval_map[name](relation, options = options, primary = primary))

        return evaluations

def main(arguments):
    options = {}
    if (arguments.options is not None):
        for (key, value) in arguments.options:
            options[key] = value

    engine_type = srli.engine.load(srli.engine.Engine(arguments.engine))

    pipeline = Pipeline.from_psl_config(arguments.config_path)
    print(pipeline)
    pipeline.run(engine_type, additional_options = options)

def _load_args():
    parser = argparse.ArgumentParser(description = 'Run a SRLi pipeline from a PSL-style config file.')

    parser.add_argument('config_path',
        action = 'store', type = str,
        help = 'The path the the PSL-style JSON config.')

    parser.add_argument('--engine', dest = 'engine',
        action = 'store', type = str, default = srli.engine.Engine.PSL.name,
        choices = [engine_type.name for engine_type in srli.engine.Engine],
        help = 'The engine to run the pipeline with.')

    parser.add_argument('--option', dest = 'options',
        action = 'append', type = str, nargs = 2,
        metavar=('key', 'value'),
        help = 'Additional options to pass to the engine.')

    arguments = parser.parse_args()

    return arguments

if (__name__ == '__main__'):
    main(_load_args())
