import os

import jpype
import jpype.imports
import jpype.types

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
CLASSPATH = os.path.join(THIS_DIR, 'jars', '*')

# TODO(eriq): Because of limitations in jpype, the JVM cannot be shutdown and started again.

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

    ground_rules = grounding_api.ground(rules, relationNames, relationArities, observed_data, unobserved_data)
    ground_rules = [GroundRuleInfo(ground_rule) for ground_rule in ground_rules]

    return ground_rules

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

def _shutdown():
    jpype.shutdownJVM()

def _init():
    if (not jpype.isJVMStarted()):
        jpype.startJVM(classpath = [CLASSPATH])

    from org.linqs.psl.java import GroundingAPI
    return GroundingAPI

class GroundRuleInfo(object):
    """
    A Python mirror of the Java class.
    With this, the JVM can be shutdown before the rules are passed to the user.
    """

    def __init__(self, java_ground_rule_info):
        self.ruleIndex = int(java_ground_rule_info.ruleIndex)
        self.operator = str(java_ground_rule_info.operator)
        self.constant = float(java_ground_rule_info.constant)
        self.coefficients = list(map(float, java_ground_rule_info.coefficients))
        self.atoms = list(map(int, java_ground_rule_info.atoms))
