import sys

import lark

CONJUNCTION = 'AND'
DISJUNCTION = 'OR'

KEY_OPERATION = 'operation'
KEY_NEGATED = 'negated'
KEY_PREDICATE = 'predicate'
KEY_ARGUMENTS = 'arguments'

GRAMMAR = '''
    %import common.CNAME
    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS

    %ignore WS

    ?rule: implication

    implication: conjunction _THEN disjunction

    ?conjunction: conjunct
                | conjunction _AND conjunct

    ?conjunct: atom_expression
             | _LPAREN conjunction _RPAREN

    ?disjunction: disjunct
                | disjunction _OR disjunct

    ?disjunct: atom_expression
             | _LPAREN disjunction _RPAREN

    ?atom_expression: atom
                    | NOT atom_expression -> negation
                    | _LPAREN atom_expression _RPAREN

    atom: predicate _LPAREN term (_COMMA term)* _RPAREN

    ?predicate: IDENTIFIER

    ?term: variable
         | constant

    ?variable: IDENTIFIER

    ?constant: ESCAPED_STRING

    IDENTIFIER: CNAME

    _AND    : "&"
    _COMMA  : ","
    _LPAREN : "("
    NOT    : "!"
    _OR     : "|"
    _RPAREN : ")"
    _THEN   : "->"
'''

class Implication(tuple):
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Implication(' + ', '.join(map(repr, self)) + ')'

class Conjunction(tuple):
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Conjunction(' + ', '.join(map(repr, self)) + ')'

class Disjunction(tuple):
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Disjunction(' + ', '.join(map(repr, self)) + ')'

class Atom(dict):
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Atom{' + ', '.join([str(key) + ':' + str(value) for (key, value) in self.items()]) + '}'

class CleanTree(lark.Transformer):
    def implication(self, elements):
        return Implication([elements[0], elements[1]])

    def conjunction(self, elements):
        return Conjunction(list(elements))

    def disjunction(self, elements):
        return Disjunction(list(elements))

    def atom(self, elements):
        return Atom({
            KEY_NEGATED: False,
            KEY_PREDICATE: str(elements[0]),
            KEY_ARGUMENTS: tuple(map(str, elements[1:])),
        })

    def negation(self, elements):
        atom = elements[1]
        atom[KEY_NEGATED] = True
        return atom

def parse(rule):
    parser = lark.Lark(GRAMMAR, start = 'rule', parser = 'lalr')
    ast = parser.parse(rule)
    cleanAST = CleanTree().transform(ast)
    return cleanAST

def main(rule):
    print(parse(rule))

def _load_args(args):
    executable = args.pop(0)
    if (len(args) == 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <rule>" % (executable), file = sys.stderr)
        sys.exit(1)

    return ' '.join(args)

if (__name__ == '__main__'):
    main(_load_args(sys.argv))
