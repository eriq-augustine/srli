import os

import jpype
import jpype.imports
import jpype.types

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
CLASSPATH = os.path.join(THIS_DIR, 'jars', '*')

def ground(relations, rules):
    grounding_api = _init()

    rules = _convert_rules(rules)
    relationNames, relationArities = _convert_relations(relations)

    # {relationName: [[arg1, ...], ...], ...}
    observed_data = jpype.java.util.HashMap()
    unobserved_data = jpype.java.util.HashMap()

    for relation in relations:
        if (relation.has_observed_data()):
            data = jpype.java.util.ArrayList()
            for row in relation.get_observed_data():
                data.add(jpype.java.util.ArrayList(row))
            observed_data[relation.name()] = data

        if (relation.has_unobserved_data()):
            data = jpype.java.util.ArrayList()
            for row in relation.get_unobserved_data():
                data.add(jpype.java.util.ArrayList(row))
            unobserved_data[relation.name()] = data

    groundrules = grounding_api.ground(relationNames, relationArities, rules, observed_data, unobserved_data)

    # TEST
    print('TEST 5 - ', len(groundrules))
    for groundrule in groundrules:
        print(groundrule)

    # TEST
    return None

def _convert_relations(relations):
    relationNames = jpype.java.util.ArrayList()
    relationArities = jpype.java.util.ArrayList()

    for relation in relations:
        relationNames.add(relation.name())
        relationArities.add(jpype.java.lang.Integer(relation.arity()))

    return relationNames, relationArities

# Convert rules into something PSL understands.
def _convert_rules(rules):
    return jpype.java.util.ArrayList([rule + ' .' for rule in rules])

def _init():
    jpype.startJVM(classpath = [CLASSPATH])
    from org.linqs.psl.java import GroundingAPI
    return GroundingAPI
