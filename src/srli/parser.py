import sys

import lark

import srli.pipeline

GRAMMAR = '''
    %import common.CNAME
    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS

    %ignore WS

    ?rule: implication
         | linear_relation

    // Logical

    implication: conjunction _THEN disjunction

    ?conjunction: conjunct
                | conjunction _AND conjunct

    ?conjunct: atom_expression
             | _LPAREN conjunction _RPAREN

    ?disjunction: disjunct
                | disjunction _OR disjunct

    ?disjunct: atom_expression
             | _LPAREN disjunction _RPAREN

    // Arithmetic

    linear_relation: linear_combination relational_op linear_combination

    ?linear_combination: atom_value
                       | linear_combination addition_op atom_value

    atom_value: atom
              | SIGNED_NUMBER

    ?relational_op: EQ

    ?addition_op: ADD
                | MINUS

    // Atom

    ?atom_expression: atom
                    | NOT atom_expression -> negation
                    | _LPAREN atom_expression _RPAREN
                    | term_operation

    ?term_operation: _LPAREN term NEQ term _RPAREN
                   |         term NEQ term

    atom: relation _LPAREN term (_COMMA term)* _RPAREN

    ?relation: IDENTIFIER

    term: variable
        | constant

    ?variable: IDENTIFIER

    ?constant: _SQUOTE /[^']+/ _SQUOTE
             | _DQUOTE /[^"]+/ _DQUOTE

    IDENTIFIER: CNAME

    ADD     : "+"
    _AND    : "&"
    _COMMA  : ","
    EQ      : "==" | "="
    _LPAREN : "("
    MINUS   : "-"
    NEQ     : "!="
    NOT     : "!" | "~"
    _OR     : "|"
    _RPAREN : ")"
    _THEN   : "->" | ">>"
    _SQUOTE : "'"
    _DQUOTE : "\\""
'''

class DNF(object):
    def __init__(self, components):
        self.atoms = []
        self.term_operations = []

        for component in components:
            if (isinstance(component, Atom)):
                self.atoms.append(component)
            elif (isinstance(component, TermOperation)):
                self.term_operations.append(component)
            else:
                raise ValueError("Unknown DNF component type (%s): %s." % (type(component), component))

    def get_atoms(self):
        return self.atoms

    def __repr__(self):
        return "%s :: {%s}" % (' | '.join(map(str, self.atoms)), ', '.join(map(str, self.term_operations)))

class LinearRelation(object):
    """
    SUM(|atoms|) |operator| |constant|
    """
    def __init__(self, operator, constant, components):
        self.operator = operator
        self.constant = constant

        self.atoms = []
        self.term_operations = []

        for component in components:
            if (isinstance(component, Atom)):
                self.atoms.append(component)
            elif (isinstance(component, TermOperation)):
                self.term_operations.append(component)
            else:
                raise ValueError("Unknown LinearRelation component type (%s): %s." % (type(component), component))

    def get_atoms(self):
        return self.atoms

    def __repr__(self):
        return "%s %s %f :: {%s}" % (' + '.join(map(str, self.atoms)), self.operator, self.constant, ', '.join(map(str, self.term_operations)))

class Atom(object):
    def __init__(self, relation_name, arguments, modifier = 1, logical = True):
        self.relation_name = relation_name
        self.relation = None
        self.arguments = arguments
        self.modifier = modifier
        self.logical = logical

    def flip(self):
        self.modifier = -self.modifier
        return self

    def __repr__(self):
        modifier = ''
        if (self.logical and (self.modifier < 0)):
            modifier = '!'
        elif ((not self.logical) and (self.modifier != 1)):
            modifier = str(self.modifier) + ' * '

        return "%s%s(%s)" % (modifier, self.relation_name, ', '.join(self.arguments))

class TermOperation(object):
    def __init__(self, operator, arguments):
        self.operator = operator
        self.arguments = arguments
        self.logical = False

    def flip(self):
        # These operations are never flipped.
        return self

    def __repr__(self):
        return "(%s %s %s)" % (self.arguments[0], self.operator, self.arguments[1])

class Variable(str):
    pass

class Constant(str):
    def __str__(self):
        return "'" + self.replace("'", "\\'") + "'"

def _make_list(element):
    if (isinstance(element, list)):
        return list(element)

    return [element]

class CleanTree(lark.Transformer):
    # Logical

    def implication(self, elements):
        conjuncts = _make_list(elements[0])
        disjuncts = _make_list(elements[1])

        for conjunct in conjuncts:
            disjuncts.append(conjunct.flip())

        return DNF(disjuncts)

    # Return: [atom, ...]
    def conjunction(self, elements):
        conjuncts = []

        for element in elements:
            if (isinstance(element, list)):
                for subelement in element:
                    conjuncts.append(subelement)
            else:
                conjuncts.append(element)

        return conjuncts

    # Return: [atom, ...]
    def disjunction(self, elements):
        disjuncts = []

        for element in elements:
            if (isinstance(element, list)):
                for subelement in element:
                    disjuncts.append(subelement)
            else:
                disjuncts.append(element)

        return disjuncts

    # Arithmetic

    # TODO(eriq): Fails with constants. Move constands to RHS and atoms to LHS.
    def linear_relation(self, elements):
        # Move all atoms to the LHS and constants to the RHS (constant).

        lhs = _make_list(elements[0])
        rhs = _make_list(elements[2])

        components = []
        operator = str(elements[1])
        constant = 0.0

        for component in lhs:
            if (isinstance(component, float) or isinstance(component, lark.lexer.Token)):
                constant -= float(str(component))
            else:
                components.append(component)

        for component in rhs:
            if (isinstance(component, float) or isinstance(component, lark.lexer.Token)):
                constant += float(str(component))
            else:
                components.append(component.flip())

        for component in components:
            component.logical = False

        return LinearRelation(operator, constant, components)

    def linear_combination(self, elements):
        operation = str(elements[1])

        lhs = _make_list(elements[0])
        rhs = _make_list(elements[2])

        if (operation == '-'):
            rhs = [component.flip() for component in rhs]

        components = lhs + rhs

        for component in components:
            component.logical = False

        return components

    # Atoms

    def atom(self, elements):
        return Atom(str(elements[0]), tuple(map(str, elements[1:])))

    def atom_value(self, elements):
        if (isinstance(elements[0], Atom)):
            return elements[0]

        # A numeric token.
        return float(str(elements[0]))

    def term_operation(self, elements):
        return TermOperation(str(elements[1]), (str(elements[0]), str(elements[2])))

    def negation(self, elements):
        atom = elements[1]
        return atom.flip()

    def term(self, elements):
        if (elements[0].type == 'IDENTIFIER'):
            return Variable(str(elements[0]))

        return Constant(str(elements[0]))

def parse(rule):
    parser = lark.Lark(GRAMMAR, start = 'rule', parser = 'lalr')

    try:
        ast = parser.parse(rule)
    except Exception as ex:
        print("Failed to parse rule: '%s'." % (rule))
        raise ex

    cleanAST = CleanTree().transform(ast)

    return cleanAST

def main(path):
    pipeline = srli.pipeline.Pipeline.from_psl_config(path)

    for rule in pipeline._rules:
        print("Input: " + rule.text())
        print("Output: " + str(parse(rule.text())))

def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <json config path>" % (executable), file = sys.stderr)
        sys.exit(1)

    return args.pop(0)

if (__name__ == '__main__'):
    main(_load_args(sys.argv))
