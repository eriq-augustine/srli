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

    # [relationIndex][row][arg]
    observed_data = []
    unobserved_data = []

    for relation in relations:
        observed_data.append(relation.get_observed_data())
        unobserved_data.append(relation.get_unobserved_data())

    groundrules = grounding_api.ground(rules, relationNames, relationArities, observed_data, unobserved_data)

    # TEST
    print('TEST 5 - ', len(groundrules))
    for groundrule in groundrules:
        print(groundrule)

    # TEST
    return None

def _convert_relations(relations):
    relationNames = []
    relationArities = []

    for relation in relations:
        relationNames.append(relation.name())
        relationArities.append(relation.arity())

    return relationNames, relationArities

# Convert rules into something PSL understands.
def _convert_rules(rules):
    return [rule + ' .' for rule in rules]

def _init():
    jpype.startJVM(classpath = [CLASSPATH])
    from org.linqs.psl.java import GroundingAPI
    return GroundingAPI
