import json

DEFAULT_TRUTH_THRESHOLD = 0.5


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

    if (len(expected) > len(predicted)):
        raise ValueError("Expecting (%d) more values than actually predicted (%d)." % (len(expected), len(predicted)))

    # If the sizes don't match, then that means there may be latent or missing variables.
    output_warning = False
    if (len(expected) != len(predicted)):
        new_expected = []
        new_predicted = []

        expected_map = {tuple(row[0:-1]) : float(row[-1]) for row in expected}
        predicted_map = {tuple(row[0:-1]) : float(row[-1]) for row in predicted}

        for key in expected_map:
            if (key not in predicted_map):
                if (not output_warning):
                    print("WARNING: Atom(s) for the %s relation were found in truth (len: %d) that were not in the predictions (len: %d). Example: %s(%s)." % (relation.name(), len(expected), len(predicted), relation.name(), ', '.join(map(str, key))))
                    output_warning = True

                continue

            new_expected.append(expected_map[key])
            new_predicted.append(predicted_map[key])

        expected = new_expected
        predicted = new_predicted
    else:
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

def load_json_with_comments(path):
    contents = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if (line == '' or line.startswith('#') or line.startswith('//')):
                continue

            if ('/*' in line):
                raise ValueError("Multi-line comments ('/* ... */') not allowed.")

            contents.append(line)

    return json.loads(' '.join(contents))
