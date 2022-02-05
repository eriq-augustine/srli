import os

import tests.base
import tests.data.simpleacquaintances.model

class TestModel(tests.base.BaseTest):
    def test_simple_acquaintances(self):
        results = tests.data.simpleacquaintances.model.run()

        self.assertEquals(len(results), 1)

        relation, data = list(results.items())[0]
        self.assertEquals(relation.name(), 'KNOWS')
        self.assertEquals(len(data), 52)
