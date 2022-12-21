import argparse
import json
import os
import re
import time

import sklearn.metrics

import srli.engine.psl.engine
import srli.relation
import srli.rule
import srli.util

class Pipeline(object):
    """
    A container that describes how to define, run, and evaluate a SRLi model.
    Currently, the capabilities of describing, building, and running a pipeline are very limited (as a prototype).
    """

    def __init__(self, options, rules, relations, learn_data, infer_data, evaluations):
        self._options = options
        self._rules = rules
        self._relations = relations
        self._evaluations = evaluations

        self._learn_data = learn_data
        self._infer_data = infer_data

    def run(self, engine_type, additional_options = {}):
        options = dict(self._options)
        options.update(additional_options)

        engine = engine_type(self._relations, self._rules,
                options = options, evaluations = self._evaluations)

        if ((len(self._learn_data) > 0) and (self._learn_data != self._infer_data)):
            self._learn(engine)

        if (len(self._infer_data) > 0):
            self._infer(engine)
            pass

    def __repr__(self):
        return json.dumps({
            'options': self._options,
            'rules': [rule.to_dict() for rule in self._rules],
            'relations': [relation.to_dict() for relation in self._relations],
            'learn_data': {str(relation) : {str(data_type) : data_sources for (data_type, data_sources) in data_spec.items()} for (relation, data_spec) in self._learn_data.items()},
            'infer_data': {str(relation) : {str(data_type) : data_sources for (data_type, data_sources) in data_spec.items()} for (relation, data_spec) in self._infer_data.items()},
            'evaluations': [evaluation.to_dict() for evaluation in self._evaluations],
        }, indent = 4)

    def _learn(self, engine):
        print("%d -- Loading learning data." % (int(time.time())))

        for relation in self._relations:
            relation.clear_data()

        for (relation, data) in self._learn_data.items():
            for (data_type, data_sources) in data.items():
                for path in data_sources['paths']:
                    relation.add_data_file(path, data_type = data_type)

                relation.add_data(data = data_sources['points'], data_type = data_type)

        print("%d -- Starting learning engine." % (int(time.time())))

        engine.learn()

    def _infer(self, engine):
        print("%d -- Loading inference data." % (int(time.time())))

        for relation in self._relations:
            relation.clear_data()

        for (relation, data) in self._infer_data.items():
            for (data_type, data_sources) in data.items():
                for path in data_sources['paths']:
                    relation.add_data_file(path, data_type = data_type)

                relation.add_data(data = data_sources['points'], data_type = data_type)

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
        config = srli.util.load_json_with_comments(path)
        base_path = os.path.dirname(path)

        options = Pipeline._parse_options(config)
        relations, learn_data, infer_data, evaluations = Pipeline._parse_relations(config, base_path)
        rules = Pipeline._parse_rules(config, relations)

        return Pipeline(options, rules, relations, learn_data, infer_data, evaluations)

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

        relation_map = {relation.name().upper() : relation for relation in relations}

        for base_rule in base_rules:
            base_rule = re.sub(r'\s+', ' ', base_rule).strip()

            match = re.search(r'^(\d+(?:\.\d+)?)\s*:\s*(.+?)\s*((?:\^[12])?)$', base_rule)
            if (match is not None):
                # A weighted rule.

                base_rule = match.group(2).strip()
                weight = float(match.group(1))
                modifier = match.group(3)

                # Check for a prior
                match = re.search(r'^[!~](\w+)\([^\)]+\)$', base_rule)
                if (match is not None):
                    name = match.group(1).upper()
                    relation_map[name].set_negative_prior_weight(weight)
                    continue

                # Check for a sum constraint.
                if (Pipeline._parse_sum_constraint(base_rule, relation_map, weight)):
                    continue

                rules.append(srli.rule.Rule(base_rule, weight = weight, squared = (modifier == '^2')))

                continue

            match = re.search(r'^(.+?)\s*\.$', base_rule)
            if (match is not None):
                # An unweighted rule.

                base_rule = match.group(1).strip()

                # Check for a sum constraint.
                if (Pipeline._parse_sum_constraint(base_rule, relation_map, None)):
                    continue

                rules.append(srli.rule.Rule(base_rule))

                continue

            raise ValueError("Could not parse rule: [%s]." % (base_rule))

        return rules

    @staticmethod
    def _parse_sum_constraint(base_rule, relation_map, weight):
        """
        Return True if a summation constraint was parsed and added to the relevant relation.
        False otherwise.
        """

        match = re.search(r'([^\(]+)\(\s*([^\)]+)\s*\)\s*([<>=]+)\s*(\d+(?:\.\d+)?)$', base_rule.upper())
        if (match is None):
            return False

        name = match.group(1).upper()
        raw_args = match.group(2)
        comparison = srli.relation.Relation.SumConstraint.SumConstraintComparison(match.group(3))
        constant = float(match.group(4))

        label_indexes = []
        split_args = [arg.strip() for arg in raw_args.split(',')]
        for i in range(len(split_args)):
            if (split_args[i].startswith('+')):
                label_indexes.append(i)

        sum_constraint = srli.relation.Relation.SumConstraint(label_indexes = label_indexes,
                comparison = comparison, constant = constant, weight = weight)

        relation_map[name].set_sum_constraint(sum_constraint)

        return True

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

        for (partition, raw_sources) in data_config.items():
            paths = []
            points = []

            for source in raw_sources:
                if (isinstance(source, str)):
                    if (not os.path.isabs(source)):
                        source = os.path.join(base_path, source)
                    paths.append(os.path.normpath(source))
                elif (isinstance(source, list)):
                    points.append(source)
                else:
                    raise ValueError("Unknown type (%s) for data source: '%s'." % (type(source), source))

            for (key, sources) in [('paths', paths), ('points', points)]:
                if (partition == 'learn'):
                    Pipeline._add_data_spec(learn_data, data_type, sources, key)
                elif (partition == 'infer'):
                    Pipeline._add_data_spec(infer_data, data_type, sources, key)
                elif (partition == 'all'):
                    Pipeline._add_data_spec(learn_data, data_type, sources, key)
                    Pipeline._add_data_spec(infer_data, data_type, sources, key)
                else:
                    raise ValueError("Unknown data phase: '%s'." % (partition))

    @staticmethod
    def _add_data_spec(data, data_type, new_list, key):
        if (key not in ['paths', 'points']):
            raise ValueError("Unknown data key ('%s')." % (key))

        if (data_type not in data):
            data[data_type] = {'paths': [], 'points': []}

        data[data_type][key] += new_list

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
