import os

import tests.base
import tests.data.simpleacquaintances.model

class TestModel(tests.base.BaseTest):
    def test_simple_acquaintances_file(self):
        results = tests.data.simpleacquaintances.model.run(load_data_files = True)

        self.assertEquals(len(results), 1)

        relation, data = list(results.items())[0]
        self.assertEquals(relation.name(), 'Knows')
        self.assertEquals(len(data), 52)

    def test_simple_acquaintances_list(self):
        results = tests.data.simpleacquaintances.model.run(load_data_files = False)

        self.assertEquals(len(results), 1)

        relation, data = list(results.items())[0]
        self.assertEquals(relation.name(), 'Knows')
        self.assertEquals(len(data), 52)

    def test_simple_acquaintances_mln(self):
        results = tests.data.simpleacquaintances.model.run(use_psl = False)

        self.assertEquals(len(results), 1)

        relation, data = list(results.items())[0]
        self.assertEquals(relation.name(), 'Knows')
        self.assertEquals(len(data), 52)
