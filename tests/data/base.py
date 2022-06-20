import abc
import csv
import os

DELIMITER = "\t"
DEFAULT_TRUTH_THRESHOLD = 0.5

class TestModel(abc.ABC):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @abc.abstractmethod
    def run(self, engine_type):
        pass

    @abc.abstractmethod
    def expected_results(self):
        pass

    def read_data_file(self, relpath):
        rows = []
        with open(os.path.join(self.data_dir, path), 'r') as file:
            for row in csv.reader(file, delimiter = DELIMITER):
                rows.append(row)
        return rows

    def load_data(self, relation, observed = [], unobserved = [], truth = []):
        for (data_type, relpaths) in [('observed', observed), ('unobserved', unobserved), ('truth', truth)]:
            for relpath in relpaths:
                relation.add_data(data_type = data_type, path = os.path.join(self.data_dir, relpath))

    def main(self):
        results, metrics = self.run()

        for (relation, data) in results.items():
            print("--- %s ---" % (relation.name()))
            for row in data:
                print("\t".join(map(str, row)))

    # TODO(eriq): There are some assumptions made here about the format of the data (e.g. a truth column at the end).
    def get_eval_data(self, results, discretize = False, truth_threshold = DEFAULT_TRUTH_THRESHOLD):
        eval_data = {}

        for relation in results:
            # Order results/truth to match.
            expected = list(sorted([list(map(str, row)) for row in relation.get_truth_data()]))
            if (len(expected) == 0):
                continue

            predicted = list(sorted([list(map(str, row)) for row in results[relation]]))

            assert len(expected) == len(predicted), ("%d vs %d" % (len(expected), len(predicted)))

            # Once sorted, just get the values.
            expected = [float(row[-1]) for row in expected]
            predicted = [float(row[-1]) for row in predicted]

            if (discretize):
                expected = [int(value >= truth_threshold) for value in expected]
                predicted = [int(value >= truth_threshold) for value in predicted]

            eval_data[relation] = {
                'expected': expected,
                'predicted': predicted,
            }

        return eval_data
