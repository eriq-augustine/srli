import srli.engine.problog.base

class ProbLog(srli.engine.problog.base.BaseGroundProbLog):
    """
    A basic implementation that builds a problog model and calls into the Python interface.
    """

    def __init__(self, relations, rules, **kwargs):
        super().__init__(relations, rules, **kwargs)

    def solve(self, **kwargs):
        atoms, ground_rules, atom_uses, sum_constraints = self._prep()

        query_atom_ids = []
        observed_atom_ids = []

        for (atom_id, atom) in atoms.items():
            if (atom.observed):
                observed_atom_ids.append(atom_id)
            else:
                query_atom_ids.append(atom_id)

        program = []

        program += self._write_ground_rules(list(range(len(ground_rules))), ground_rules, atoms, query_atom_ids, sum_constraints)
        program.append('')
        program += self._write_observations(observed_atom_ids, atoms)
        program.append('')
        program += self._write_queries(query_atom_ids, atoms)

        # print("\n".join(program))
        self._run("\n".join(program), query_atom_ids, atoms)

        return self._create_results(atoms)
