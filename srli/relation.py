import csv
import enum

class Relation(object):
    class DataType(enum.Enum):
        OBSERVED = 'observed'
        UNOBSERVED = 'unobserved'
        TRUTH = 'truth'

    # TODO(eriq): Allow more ways of specifying arguments.
    # TODO(eriq): Types are mainly ignored right now.
    def __init__(self, name, arity = None, variable_types = None):
        self._name = name
        self._arity = arity
        self._variable_types = variable_types

        if (self._arity is None and self._variable_types is not None):
            self._arity = len(self._variable_types)

        # {dataType: data, ...}
        self._data = {}
        for data_type in Relation.DataType:
            self._data[data_type] = []

        assert(self._arity is not None and self._arity > 0)

    def arity(self):
        return self._arity

    def variable_types(self):
        return self._variable_types

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
    # TODO(eriq): Check incoming data for consistency (arity, truth values, etc).

    def add_data(self, data = None, data_type = DataType.OBSERVED, path = None):
        data_type = Relation.DataType(data_type)

        if (data is not None and type(data) == list):
            self._data[data_type] += data
            return len(data)
        elif (path is not None):
            return self.add_data_file(path, data_type)

        raise NotImplementedError("Data loading method currently not implemented.")

    def add_data_file(self, path, data_type = DataType.OBSERVED, delimiter = "\t", **csv_args):
        data_type = Relation.DataType(data_type)

        count = 0
        with open(path, 'r') as file:
            for row in csv.reader(file, delimiter = delimiter, **csv_args):
                self._data[data_type].append(row)
                count += 1

        return count

    def __repr__(self):
        return "%s/%d" % (self._name, self._arity)
