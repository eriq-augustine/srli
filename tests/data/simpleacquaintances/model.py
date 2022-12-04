#!/usr/bin/env python3

import os

import sklearn.metrics

import srli.engine.mln
import srli.engine.psl
import srli.relation
import srli.util

import tests.data.base

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'data')

ENGINE_OPTIONS = {
    srli.engine.mln.NativeMLN: {
        'max_flips': 150
    }
}

class SimpleAcquaintancesModel(tests.data.base.TestModel):
    def __init__(self):
        super().__init__(DATA_DIR)

    def run(self, engine_type = srli.engine.psl.PSL):
        lived = srli.relation.Relation('Lived', arity = 2, variable_types = ['Person', 'Location'])
        likes = srli.relation.Relation('Likes', arity = 2, variable_types = ['Person', 'Thing'])
        knows = srli.relation.Relation('Knows', arity = 2, variable_types = ['Person', 'Person'],
                negative_prior_weight = 0.05)

        self.load_data(lived, observed = ['lived_obs.txt'])
        self.load_data(likes, observed = ['likes_obs.txt'])
        self.load_data(knows, observed = ['knows_obs.txt'], unobserved = ['knows_targets.txt'], truth = ['knows_truth.txt'])

        rules = [
            'Lived(P1, L) & Lived(P2, L) & (P1 != P2) -> Knows(P1, P2)',
            'Lived(P1, L1) & Lived(P2, L2) & (P1 != P2) & (L1 != L2) -> !Knows(P1, P2)',
            'Likes(P1, L) & Likes(P2, L) & (P1 != P2) -> Knows(P1, P2)',
            'Knows(P1, P2) & Knows(P2, P3) & (P1 != P3) -> Knows(P1, P3)',
            'Knows(P1, P2) = Knows(P2, P1)',
        ]

        weights = [0.20, 0.05, 0.10, 0.05, None, 0.05]
        squared = [True, True, True, True, None, True]

        engine = engine_type(
                relations = [lived, likes, knows],
                rules = rules,
                # PSL-specific.
                weights = weights,
                squared = squared)

        options = {}
        if (engine_type in ENGINE_OPTIONS):
            options = ENGINE_OPTIONS[engine_type]

        results = engine.solve(**options)

        expected, predicted = srli.util.get_eval_values(knows, results[knows], discretize = True)
        f1 = sklearn.metrics.f1_score(expected, predicted)

        return results, {knows: {'f1': f1}}

    def expected_results(self):
        return {
            'Knows': {
                'size': 118,
                'min_metrics': {
                    'f1': 0.50,
                }
            }
        }

if (__name__ == '__main__'):
    model = SimpleAcquaintancesModel()
    model.run()
