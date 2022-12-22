import csv
import enum
import math
import string

MAX_ARITY = len(string.ascii_uppercase)

class Relation(object):
    class DataType(enum.Enum):
        OBSERVED = 'observed'
        UNOBSERVED = 'unobserved'
        TRUTH = 'truth'

    class SumConstraint(object):
        class SumConstraintComparison(enum.Enum):
            LT = '<'
            LTE = '<='
            EQ = '='
            GTE = '>='
            GT = '>'

        def __init__(self, label_indexes = [-1], comparison = SumConstraintComparison.EQ,
                constant = 1, weight = None):
            self.label_indexes = list(label_indexes)
            self.comparison = comparison
            self.constant = constant
            self.weight = weight

        def resolve_indexes(self, arity):
            """
            Resolve any negative indexes using the known arity.
            """

            for i in range(len(self._label_indexes)):
                if (self._label_indexes[i] < 0):
                    self._label_indexes[i] += arity

        def is_hard_functional(self):
            return self.is_functional() and (self.weight is None)

        def is_functional(self):
            return (self.comparison == self.SumConstraintComparison.EQ) and math.isclose(1.0, self.constant)

        def to_dict(self):
            return {
                'label_indexes': self.label_indexes,
                'comparison': self.comparison.value,
                'constant': self.constant,
                'weight': self.weight,
            }

    # TODO(eriq): Allow more ways of specifying arguments.
    # TODO(eriq): Types are mainly ignored right now.
    # TODO(eriq): Priors can be much more expressive and complicated.
    def __init__(self, name, arity = None, variable_types = None,
            negative_prior_weight = None, sum_constraint = None):
        self._name = name
        self._arity = arity
        self._variable_types = variable_types
        self._negative_prior_weight = negative_prior_weight
        self._sum_constraint = sum_constraint

        if ((self._arity is None) and (self._variable_types is not None)):
            self._arity = len(self._variable_types)

        if ((self._arity is None) or (self._arity <= 0)):
            raise ValueError("Arity for relation (%s) must be > 0, found: %s.", self._name, str(self._arity))

        if (self._arity > MAX_ARITY):
            raise ValueError("%s -- Relation arity too large, must be <= %d." % (str(self), MAX_ARITY))

        if ((self._variable_types is not None) and (self._arity != len(self._variable_types))):
            raise ValueError("Relation's (%s) arity (%d) must be consistent with length of variables types (%d)." % (self._name, self._arity, len(self._variable_types)))

        self.clear_data()

    def arity(self):
        return self._arity

    def variable_types(self):
        return self._variable_types

    def set_variable_types(self, variable_types):
        self._variable_types = variable_types

    def name(self):
        return self._name

    def is_observed(self):
        return self.has_observed_data() and not self.has_unobserved_data()

    def sum_constraint(self):
        return self._sum_constraint

    def has_sum_constraint(self):
        return self._sum_constraint is not None

    def set_sum_constraint(self, sum_constraint):
        self._sum_constraint = sum_constraint

    def has_negative_prior_weight(self):
        return (self._negative_prior_weight is not None)

    def get_negative_prior_weight(self):
        return self._negative_prior_weight

    def set_negative_prior_weight(self, weight):
        self._negative_prior_weight = weight

    def has_data(self, data_type):
        return len(self._data[data_type]) != 0

    def has_observed_data(self):
        return self.has_data(Relation.DataType.OBSERVED)

    def has_unobserved_data(self):
        return self.has_data(Relation.DataType.UNOBSERVED)

    def has_truth_data(self):
        return self.has_data(Relation.DataType.TRUTH)

    def get_data(self, data_type):
        return self._data[data_type]

    def get_observed_data(self):
        return self.get_data(Relation.DataType.OBSERVED)

    def get_unobserved_data(self):
        return self.get_data(Relation.DataType.UNOBSERVED)

    def get_truth_data(self):
        return self.get_data(Relation.DataType.TRUTH)

    def clear_data(self):
        # {dataType: data, ...}
        self._data = {}
        for data_type in Relation.DataType:
            self._data[data_type] = []

    # TODO(eriq): So much with data loading in general.
    # TODO(eriq): Check incoming data for consistency (arity, truth values, etc).

    def add_observed_data(self, data = None, path = None):
        return self.add_data(data = data, path = path, data_type = Relation.DataType.OBSERVED)

    def add_unobserved_data(self, data = None, path = None):
        return self.add_data(data = data, path = path, data_type = Relation.DataType.UNOBSERVED)

    def add_truth_data(self, data = None, path = None):
        return self.add_data(data = data, path = path, data_type = Relation.DataType.TRUTH)

    def add_data(self, data = None, data_type = DataType.OBSERVED, path = None):
        data_type = Relation.DataType(data_type)

        if ((data is not None) and (path is not None)):
            raise NotImplementedError("Loading both local and file data at the same time not implemented.")

        if (data is not None and type(data) == list):
            self._data[data_type] += data
            return len(data)
        elif (path is not None):
            return self.add_data_file(path, data_type)

        raise NotImplementedError("Data loading method currently not implemented.")

    def add_data_file(self, path, data_type = DataType.OBSERVED, delimiter = "\t", **csv_args):
        data_type = Relation.DataType(data_type)

        if ('quoting' not in csv_args):
            csv_args['quoting'] = csv.QUOTE_NONE

        count = 0
        with open(path, 'r') as file:
            for row in csv.reader(file, delimiter = delimiter, **csv_args):
                self._data[data_type].append(row)
                count += 1

        return count

    def __repr__(self):
        return "%s/%d" % (self._name, self._arity)

    def to_dict(self):
        rtn = {
            'name': self._name,
            'arity': self._arity,
        }

        if (self._negative_prior_weight is not None):
            rtn['negative_prior_weight'] = self._negative_prior_weight

        if (self._sum_constraint):
            rtn['sum_constraint'] = self._sum_constraint.to_dict()

        if (self._variable_types is not None):
            rtn['variable_type'] = self._variable_types

        return rtn
