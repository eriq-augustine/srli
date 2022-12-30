import math

import pysat.examples.rc2
import pysat.formula

import srli.engine.mln.base

class PySATMLN(srli.engine.mln.base.BaseMLN):
    """
    A basic implementation of MLNs with inference using PySAT as a SAT solver.
    """

    def __init__(self, relations, rules, **kwargs):
        super().__init__(relations, rules, **kwargs)

    def reason(self, ground_rules, atoms, **kwargs):
        ground_rules, atoms = self._adjust_atom_ids(ground_rules, atoms)

        cnf = self._create_cnf(ground_rules, atoms)
        rc2 = pysat.examples.rc2.RC2Stratified(cnf, solver = 'Gluecard4',
                adapt = True, exhaust = True, minz = True, trim = 10)
        solution = rc2.compute()

        # Construct the results: {atom_id: value, ...}.
        # Remember to re-adjust the atom ids.
        return {(abs(atom_id) - 1) : (0.0 if atom_id < 0.0 else 1.0) for atom_id in solution}

    # PySat does not allow 0 for an id, so we need to add 1 to all atom ids.
    def _adjust_atom_ids(self, ground_rules, atoms):
        for ground_rule in ground_rules:
            ground_rule.atoms = [atom_id + 1 for atom_id in ground_rule.atoms]

        new_atoms = {atom_id + 1 : atom for (atom_id, atom) in atoms.items()}

        return ground_rules, new_atoms

    def _create_cnf(self, ground_rules, atoms):
        cnf = pysat.formula.WCNFPlus()

        # Add in priors.
        for (atom_index, atom) in atoms.items():
            if (not atom['observed'] and atom['relation'].has_negative_prior_weight()):
                cnf.append([-atom_index], weight = atom['relation'].get_negative_prior_weight())

        # Add actual ground rules.
        for ground_rule in ground_rules:
            if (ground_rule.operator == '|'):
                self._convert_logical_rule(cnf, ground_rule, atoms)
            elif (ground_rule.operator == '='):
                self._convert_arithmetic_rule(cnf, ground_rule, atoms)
            else:
                raise ValueError("Unsupported MLN rule operator: '%s'." % (ground_rule.operator))

        return cnf

    def _convert_logical_rule(self, cnf, ground_rule, ground_atoms):
        rule = self._rules[ground_rule.rule_index]
        weight = ground_rule.weight

        atoms = ground_rule.atoms
        constant = ground_rule.constant

        terms = []

        for i in range(len(atoms)):
            atom_id = atoms[i]
            atom = ground_atoms[atom_id]

            coefficient = int(ground_rule.coefficients[i])
            if (coefficient == 0):
                continue

            if (coefficient not in [-1, 1]):
                raise ValueError("MLN can only have logical rules with coefficients in {-1, 1}, found: %d,  [%s]." % (coefficient, rule.text()))

            # If observed, this term may be trivial.
            value = atom['value']
            if (atom['observed']):
                if ((coefficient == 1 and math.isclose(value, 1.0)) or (coefficient == -1 and math.isclose(value, 0.0))):
                    return
                continue

            # Since we are swapping DNF to CNF, flip the sign on a negative coefficient.
            # Note that we are both swapping from a DNF to CNF and also min loss to max sat.
            # So, signs carry over.
            if (coefficient == -1):
                atom_id *= -1

            terms.append(atom_id)

        cnf.append(terms, weight = weight)

    # TODO(eriq): We can support a broader range of arithmetic rules, if we go deeper in the analysis.
    # Assumes all coefficients are integers.
    def _convert_arithmetic_rule(self, cnf, ground_rule, ground_atoms):
        rule = self._rules[ground_rule.rule_index]
        weight = ground_rule.weight

        atoms, coefficients, constant = self._collapse_arithemtic(ground_rule, ground_atoms)

        if (len(atoms) == 0):
            return

        if (len(atoms) not in [1, 2]):
            raise ValueError("MLN arithmetic rules can only have one or two atoms, found %d, [%s]." % (len(atoms), rule.text()))

        for i in range(len(coefficients)):
            if (abs(coefficients[i]) != 1):
                raise ValueError("MLN arithmetic rules can only have a -1 or 1 coefficients, found: %f, [%s]." % (coefficients[i], rule.text()))

        if (len(atoms) == 1):
            # Check for trivial.
            if (ground_atoms[atoms[0]]['observed']):
                return

            if (constant not in [0, 1]):
                raise ValueError("MLN arithmetic rules can only have a 0.0 or 1.0 constant, found: %f, [%s]." % (constant, rule.text()))

            sign = -1
            if ((constant == 1 and coefficients[0] == 1) or (constant == 0 and coefficients[0] == -1)):
                sign = 1;

            cnf.append([sign * atoms[0]], weight = weight)
            return

        # len(atoms) == 2

        if (constant != 0):
            raise ValueError("MLN arithmetic binary rules can only have a 0.0 constant, found: %f,  [%s]." % (constant, rule.text()))

        for i in range(len(ground_rule.atoms)):
            if (ground_atoms[atoms[i]]['observed']):
                raise ValueError("Not expecting an observed atom,  [%s]." % (rule.text()))

        # If the coefficients are the same, then theese atoms should differ (they are both on the LHS).
        if (coefficients[0] == coefficients[1]):
            cnf.append([[atoms[0], atoms[1]], 1], is_atmost = True, weight = weight)
            cnf.append([[-atoms[0], -atoms[1]], 1], is_atmost = True, weight = weight)
        else:
            # Otherwise, the coefficients differ and the atoms should match.
            cnf.append([[-atoms[0], atoms[1]], 1], is_atmost = True, weight = weight)
            cnf.append([[atoms[0], -atoms[1]], 1], is_atmost = True, weight = weight)

    # Fold any observations into the constant.
    def _collapse_arithemtic(self, ground_rule, ground_atoms):
        atoms = ground_rule.atoms
        coefficients = ground_rule.coefficients
        constant = ground_rule.constant

        new_atoms = []
        new_coefficients = []

        for i in range(len(atoms)):
            atom_id = atoms[i]
            atom = ground_atoms[atom_id]
            coefficient = int(coefficients[i])

            if (math.isclose(coefficient, 0.0)):
                continue

            if (not atom['observed']):
                new_atoms.append(atom_id)
                new_coefficients.append(coefficient)
                continue

            value = int(atom.value)
            if (value == 0):
                value = -1

            constant -= coefficient * value

        if (constant < 0):
            constant = -constant
            new_coefficients = [-value for value in new_coefficients]

        return new_atoms, new_coefficients, constant
