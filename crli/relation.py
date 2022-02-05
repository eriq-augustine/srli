import csv
import enum

class Relation(object):
    class DataType(enum.Enum):
        OBSERVED = 'observed'
        UNOBSERVED = 'unobserved'
        TRUTH = 'truth'

    # TODO(eriq): Allow more ways of specifying arguments.
    def __init__(self, name, arity = None):
        self._name = name
        self._arity = arity

        # {dataType: data, ...}
        self._data = {}
        for type in Relation.DataType:
            self._data[type] = []

        assert(self._arity is not None and self._arity > 0)

    def arity(self):
        return self._arity

    def name(self):
        return self._name

    def is_observed(self):
        return self.has_observed_data() and not self.has_unobserved_data()

    def has_observed_data(self):
        return len(self._data[Relation.DataType.OBSERVED]) != 0

    def has_unobserved_data(self):
        return len(self._data[Relation.DataType.UNOBSERVED]) != 0

    def has_truth_data(self):
        return len(self._data[Relation.DataType.TRUTH]) != 0

    def get_observed_data(self):
        return self._data[Relation.DataType.OBSERVED]

    def get_unobserved_data(self):
        return self._data[Relation.DataType.UNOBSERVED]

    def get_truth_data(self):
        return self._data[Relation.DataType.TRUTH]

    # TODO(eriq): So much with data loading in general.

    def add_data(self, type = DataType.OBSERVED, path = None):
        type = Relation.DataType(type)

        if (path is not None):
            return self.add_data_file(path, type)

        raise NotImplementedError("Data loading currently needs a file.")

    def add_data_file(self, path, type = DataType.OBSERVED, delimiter = "\t", **csv_args):
        type = Relation.DataType(type)

        with open(path, 'r') as file:
            for row in csv.reader(file, delimiter = delimiter, **csv_args):
                self._data[type].append(row)
