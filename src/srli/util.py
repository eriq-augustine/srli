DEFAULT_TRUTH_THRESHOLD = 0.5

# TODO(eriq): There are some assumptions made here about the format of the data (e.g. a truth column at the end).

def get_eval_values(relation, results, discretize = False, truth_threshold = DEFAULT_TRUTH_THRESHOLD):
    """
    Take in the results from inference (for a single predicate),
    and format the data for each relation into two lists of floats representing the predicted and truth value for each data point.
    These lists should be able to be directly passed into general sklean evaluation/scoring metrics.
    Return: (expected, predicted)
    """

    # Order results/truth to match.
    expected = list(sorted([list(map(str, row)) for row in relation.get_truth_data()]))
    if (len(expected) == 0):
        return [], []

    predicted = list(sorted([list(map(str, row)) for row in results]))

    assert len(expected) == len(predicted), ("%d vs %d" % (len(expected), len(predicted)))

    # Once sorted, just get the values.
    expected = [float(row[-1]) for row in expected]
    predicted = [float(row[-1]) for row in predicted]

    if (discretize):
        expected = [int(value >= truth_threshold) for value in expected]
        predicted = [int(value >= truth_threshold) for value in predicted]

    return expected, predicted

def get_eval_categories(relation, results, label_indexes = [-1]):
    """
    Like get_eval_values(), but will get labels instead of truth values.
    Labels are identified using label_indexes (negative indexes are allowed).
    Any label that is not an index is used to identify the entity represented by the data point.
    Returns: (expected label, predicted label, entity)
    """

    label_indexes = [index if (index >= 0) else (relation.arity() + index ) for index in label_indexes]
    entity_indexes = list(sorted(set(range(relation.arity())) - set(label_indexes)))

    # {(entity): [(best label), best value], ...}
    expected_values = {}
    predicted_values = {}

    for (source, dest) in [(relation.get_truth_data(), expected_values), (results, predicted_values)]:
        for row in source:
            value = float(row[-1])
            entity = tuple([str(row[index]) for index in entity_indexes])
            label = tuple([str(row[index]) for index in label_indexes])

            if ((entity not in dest) or (value > dest[entity][1])):
                dest[entity] = [label, value]

    if (len(expected_values) != len(predicted_values)):
        print("Warning: found different sizes for truth (%d) and predictions (%d)." % (len(expected_values), len(predicted_values)))

    entities = list(sorted(set(expected_values.keys()) & set(predicted_values.keys())))

    expected = [expected_values[entity][0] for entity in entities]
    predicted = [predicted_values[entity][0] for entity in entities]

    return expected, predicted, entities
