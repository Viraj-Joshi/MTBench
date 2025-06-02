from isaacgym import gymapi, gymutil

class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed by attributes (as well
        as normally).

    >>> obj = AttrDict()
    >>> obj['test'] = 'hi'
    >>> print obj.test
    hi
    >>> del obj.test
    >>> obj.test = 'bye'
    >>> print obj['test']
    bye
    >>> print len(obj)
    1
    >>> obj.clear()
    >>> print len(obj)
    0
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})

def parse_sim_params(sim_cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if sim_cfg.physics_engine == "flex":  #  gymapi.SIM_FLEX:
        pass
    elif sim_cfg.physics_engine == "physx":  #gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = sim_cfg.use_gpu
        sim_params.physx.num_subscenes = sim_cfg.subscenes
    sim_params.use_gpu_pipeline = sim_cfg.use_gpu_pipeline

    # parse them and update/override above:
    gymutil.parse_sim_config(sim_cfg, sim_params)

    # get physics engine
    if sim_cfg.physics_engine == "flex":
        physics_engine = gymapi.SIM_FLEX
    elif sim_cfg.physics_engine == "physx":
        physics_engine = gymapi.SIM_PHYSX

    return sim_params, physics_engine

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result