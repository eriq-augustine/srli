#!/usr/bin/env python3

import os

import crli
import crli.inference
import crli.relation

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATA_DIR = os.path.join(THIS_DIR, 'data')

def run(data_dir = DEFAULT_DATA_DIR):
    lived = crli.relation.Relation('lived', arity = 2)
    likes = crli.relation.Relation('likes', arity = 2)
    knows = crli.relation.Relation('knows', arity = 2)

    lived.add_data(type = 'observed', path = os.path.join(data_dir, 'lived_obs.txt'))
    likes.add_data(type = 'observed', path = os.path.join(data_dir, 'likes_obs.txt'))

    knows.add_data(type = 'observed', path = os.path.join(data_dir, 'knows_obs.txt'))
    knows.add_data(type = 'unobserved', path = os.path.join(data_dir, 'knows_targets.txt'))
    knows.add_data(type = 'truth', path = os.path.join(data_dir, 'knows_truth.txt'))

    rules = [
        'Lived(P1, L) & Lived(P2, L) & (P1 != P2) -> Knows(P1, P2)',
        'Lived(P1, L1) & Lived(P2, L2) & (P1 != P2) & (L1 != L2) -> !Knows(P1, P2)',
        'Likes(P1, L) & Likes(P2, L) & (P1 != P2) -> Knows(P1, P2)',
        'Knows(P1, P2) & Knows(P2, P3) & (P1 != P3) -> Knows(P1, P3)',
        'Knows(P1, P2) = Knows(P2, P1)',
        '!Knows(P1, P2)'
    ]

    weights = [20.0, 5.0, 10.0, 5.0, None, 5.0]
    squared = [True, True, True, True, None, True]

    engine = crli.inference.PSL(
            relations = [lived, likes, knows],
            rules = rules,
            # PSL-specific.
            weights = weights,
            squared = squared)

    results = engine.solve()

    return results

if (__name__ == '__main__'):
    results = run()

    for (predicate, data) in results.items():
        print("--- %s ---" % (predicate.name()))
        for row in data:
            print("\t".join(map(str, row)))
