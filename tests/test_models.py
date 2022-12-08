import os

import srli.engine
import tests.base
import tests.data.simpleacquaintances.model
import tests.data.smokers.model

MODELS = [
    tests.data.simpleacquaintances.model.SimpleAcquaintancesModel,
    tests.data.smokers.model.SmokersModel,
]

SKIP_PAIRS = [
    (tests.data.simpleacquaintances.model.SimpleAcquaintancesModel, srli.engine.Engine.ProbLog),
]

class ModelTest(tests.base.BaseTest):
    pass

def _make_model_test(model_class, engine_type, additional_args = {}):
    def __test_method(self):
        model = model_class()

        expected_results = model.expected_results()
        results, metrics = model.run(engine_type = srli.engine.load(engine_type), **additional_args)

        print(metrics)

        self.assertEquals(len(results), len(expected_results))

        for (relation, data) in results.items():
            self.assertIn(relation.name(), expected_results)
            self.assertEquals(len(data), expected_results[relation.name()]['size'])

            if ('min_metrics' in expected_results[relation.name()]):
                for (metric_name, min_value) in expected_results[relation.name()]['min_metrics'].items():
                    value = metrics[relation][metric_name]
                    self.assertTrue(value >= min_value, "Metric (%s) has too low a value. Expected %fs, found %f." % (metric_name, min_value, value))

    return __test_method

for model_class in MODELS:
    for engine_type in srli.engine.Engine:
        if ((model_class, engine_type) in SKIP_PAIRS):
            continue

        test_name = "test_%s_%s" % (model_class.__name__, engine_type.name)
        test_method = _make_model_test(model_class, engine_type)

        setattr(ModelTest, test_name, test_method)
