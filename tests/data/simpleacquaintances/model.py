#!/usr/bin/env python3

import csv
import os

import srli
import srli.inference
import srli.relation

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATA_DIR = os.path.join(THIS_DIR, 'data')

DELIMITER = "\t"

def _read_file(path):
    rows = []
    with open(path, 'r') as file:
        for row in csv.reader(file, delimiter = DELIMITER):
            rows.append(row)
    return rows

def _load_data_from_files(data_dir, lived, likes, knows):
    lived.add_data(data_type = 'observed', path = os.path.join(data_dir, 'lived_obs.txt'))
    likes.add_data(data_type = 'observed', path = os.path.join(data_dir, 'likes_obs.txt'))

    knows.add_data(data_type = 'observed', path = os.path.join(data_dir, 'knows_obs.txt'))
    knows.add_data(data_type = 'unobserved', path = os.path.join(data_dir, 'knows_targets.txt'))
    knows.add_data(data_type = 'truth', path = os.path.join(data_dir, 'knows_truth.txt'))

def _load_data_from_lists(data_dir, lived, likes, knows):
    lived.add_data(data_type = 'observed', data = _read_file(os.path.join(data_dir, 'lived_obs.txt')))
    likes.add_data(data_type = 'observed', data = _read_file(os.path.join(data_dir, 'likes_obs.txt')))

    knows.add_data(data_type = 'observed', data = _read_file(os.path.join(data_dir, 'knows_obs.txt')))
    knows.add_data(data_type = 'unobserved', data = _read_file(os.path.join(data_dir, 'knows_targets.txt')))
    knows.add_data(data_type = 'truth', data = _read_file(os.path.join(data_dir, 'knows_truth.txt')))

def run(data_dir = DEFAULT_DATA_DIR, engine_type = srli.inference.PSL, load_data_files = True):
    lived = srli.relation.Relation('Lived', arity = 2, variable_types = ['Person', 'Location'])
    likes = srli.relation.Relation('Likes', arity = 2, variable_types = ['Person', 'Thing'])
    knows = srli.relation.Relation('Knows', arity = 2, variable_types = ['Person', 'Person'])

    if (load_data_files):
        _load_data_from_files(data_dir, lived, likes, knows)
    else:
        _load_data_from_lists(data_dir, lived, likes, knows)

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

    engine = engine_type(
            relations = [lived, likes, knows],
            rules = rules,
            # PSL-specific.
            weights = weights,
            squared = squared)

    results = engine.solve()

    return results

def expected_results():
    return {
        'Knows': {
            'size': 52
        }
    }

if (__name__ == '__main__'):
    results = run()

    for (predicate, data) in results.items():
        print("--- %s ---" % (predicate.name()))
        for row in data:
            print("\t".join(map(str, row)))
