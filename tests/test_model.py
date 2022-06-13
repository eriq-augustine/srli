import os

import tests.base
import tests.data.simpleacquaintances.model

TEST_MODELS = [tests.data.simpleacquaintances.model]

class TestModel(tests.base.BaseTest):
    def test_simple_acquaintances_list(self):
        args = {'load_data_files': False}
        test_method = _make_model_test(tests.data.simpleacquaintances.model, tests.base.ENGINE_PSL, additional_args = args)
        test_method(self)

def _make_model_test(model, engine, additional_args = {}):
    def __test_method(self):
        expected_results = model.expected_results()
        results = model.run(engine_type = engine, **additional_args)

        self.assertEquals(len(results), len(expected_results))

        for (relation, data) in results.items():
            self.assertIn(relation.name(), expected_results)
            self.assertEquals(len(data), expected_results[relation.name()]['size'])

    return __test_method

for model in TEST_MODELS:
    for engine in tests.base.ENGINES:
        test_name = "test_%s_%s" % (model.__name__, engine.__name__)
        test_method = _make_model_test(model, engine)

        setattr(TestModel, test_name, test_method)
