import os

import tests.base
import tests.data.simpleacquaintances.model
import tests.data.smokers.model

TEST_MODELS = [
    tests.data.simpleacquaintances.model.SimpleAcquaintancesModel,
    tests.data.smokers.model.SmokersModel,
]

SKIP_PAIRS = [
    (tests.data.simpleacquaintances.model.SimpleAcquaintancesModel, tests.base.ENGINE_PL),
]

class ModelTest(tests.base.BaseTest):
    pass

def _make_model_test(model_class, engine, additional_args = {}):
    def __test_method(self):
        model = model_class()

        expected_results = model.expected_results()
        results, metrics = model.run(engine_type = engine, **additional_args)

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

for model_class in TEST_MODELS:
    for engine in tests.base.ENGINES:
        if ((model_class, engine) in SKIP_PAIRS):
            continue

        test_name = "test_%s_%s" % (model_class.__name__, engine.__name__)
        test_method = _make_model_test(model_class, engine)

        setattr(ModelTest, test_name, test_method)
