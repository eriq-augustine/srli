import enum

class Engine(enum.Enum):
    Logic_Weighted_Discrete = 'Logic_Weighted_Discrete'
    MLN_Native = 'MLN_Native'
    MLN_PySAT = 'MLN_PySAT'
    ProbLog = 'ProbLog'
    ProbLog_NonCollective = 'ProbLog_NonCollective'
    PSL = 'PSL'
    Random_Continuous = 'Random_Continuous'
    Random_Discrete = 'Random_Discrete'
    Tuffy = 'Tuffy'

def load(engine_type):
    if (engine_type == Engine.Logic_Weighted_Discrete):
        return _load_logic_weighted_discrete()
    elif (engine_type == Engine.MLN_Native):
        return _load_mln_native()
    elif (engine_type == Engine.MLN_PySAT):
        return _load_mln_pysat()
    elif (engine_type == Engine.ProbLog):
        return _load_problog()
    elif (engine_type == Engine.ProbLog_NonCollective):
        return _load_problog_noncollective()
    elif (engine_type == Engine.PSL):
        return _load_psl()
    elif (engine_type == Engine.Random_Continuous):
        return _load_random_continuous()
    elif (engine_type == Engine.Random_Discrete):
        return _load_random_discrete()
    elif (engine_type == Engine.Tuffy):
        return _load_tuffy()

    raise ValueError("Unknown engine type: '%s'." % (engine_type))

def _load_logic_weighted_discrete():
    import srli.engine.logic.dws
    return srli.engine.logic.dws.DiscreteWeightedSolver

def _load_mln_native():
    import srli.engine.mln.native
    return srli.engine.mln.native.NativeMLN

def _load_mln_pysat():
    import srli.engine.mln.pysat
    return srli.engine.mln.pysat.PySATMLN

def _load_problog():
    import srli.engine.problog.engine
    return srli.engine.problog.engine.ProbLog

def _load_problog_noncollective():
    import srli.engine.problog.noncollective
    return srli.engine.problog.noncollective.NonCollectiveProbLog

def _load_psl():
    import srli.engine.psl.engine
    return srli.engine.psl.engine.PSL

def _load_random_continuous():
    import srli.engine.random
    return srli.engine.random.RandomContinuousEngine

def _load_random_discrete():
    import srli.engine.random
    return srli.engine.random.RandomDiscreteEngine

def _load_tuffy():
    import srli.engine.tuffy.docker
    return srli.engine.tuffy.docker.Tuffy
