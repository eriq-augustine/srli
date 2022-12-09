import abc

import sklearn.metrics

import srli.util


class Evaluation(abc.ABC):
    """
    Evaluations provide a built-in mechanism for SRLi Relations to evaluate the output of inference.
    """

    def __init__(self, metric_name, relation, options = {}, primary = False, **kwargs):
        self._metric_name = metric_name
        self._relation = relation
        self._options = options
        self._primary = primary

    @abc.abstractmethod
    def evaluate(self, results):
        """
        Take in the inference results and perform an evaluation.
        """

        pass

    def relation(self):
        return self._relation

    def metric_name(self):
        return self._metric_name

    def options(self):
        return self._options

    def is_primary(self):
        return self._primary

    def to_dict(self):
        """
        Return a simplified representation (ideal for converting to JSON).
        """

        return {
            'metric_name': self._metric_name,
            'relation': self._relation.name(),
            'primary': self._primary,
            'options': self._options,
        }

class AuPRC(Evaluation):
    def __init__(self, relation, **kwargs):
        super().__init__('AuPRC', relation, **kwargs)

    def evaluate(self, results):
        expected, predicted = srli.util.get_eval_values(self._relation, results[self._relation], discretize = False)
        return sklearn.metrics.average_precision_score(expected, predicted)

class AuROC(Evaluation):
    def __init__(self, relation, **kwargs):
        super().__init__('AuROC', relation, **kwargs)

    def evaluate(self, results):
        expected, predicted = srli.util.get_eval_values(self._relation, results[self._relation], discretize = False)
        return sklearn.metrics.roc_auc_score(expected, predicted)

class CategoricalAccuracy(Evaluation):
    def __init__(self, relation, **kwargs):
        super().__init__('Categorical Accuracy', relation, **kwargs)

    def evaluate(self, results):
        expected, predicted, _ = srli.util.get_eval_categories(self._relation, results[self._relation])
        return sklearn.metrics.accuracy_score(expected, predicted)

class RMSE(Evaluation):
    def __init__(self, relation, **kwargs):
        super().__init__('RMSE', relation, **kwargs)

    def evaluate(self, results):
        expected, predicted = srli.util.get_eval_values(self._relation, results[self._relation], discretize = False)
        return sklearn.metrics.mean_squared_error(expected, predicted, squared = False)

class F1(Evaluation):
    def __init__(self, relation, **kwargs):
        super().__init__('F1', relation, **kwargs)

    def evaluate(self, results):
        expected, predicted = srli.util.get_eval_values(self._relation, results[self._relation], discretize = True)
        return sklearn.metrics.f1_score(expected, predicted)
