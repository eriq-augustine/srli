import abc
import csv
import os

DELIMITER = "\t"

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
