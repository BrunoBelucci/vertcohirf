import yaml
import itertools
import copy


def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config

def load_split_config(args):
    with open(args.config, 'r') as f:
        whole_config = yaml.safe_load(f)

    listed_fields = []
    single_fields = {}
    for k, v in whole_config.items():
        if isinstance(v, list):
            listed_fields.append([(k, subv) for subv in v])
        else:
            single_fields[k] = v

    configs = []
    for combine in itertools.product(*listed_fields):
        config = copy.deepcopy(single_fields)
        for e in combine:
            config[e[0]] = e[1]
        configs.append(config)

    return configs

def generate_local_config(d, n ,k, eps):
    config = {}
    config['dataset'] = ""
    config['d'] = d
    config['n'] = n
    config['k'] = k
    config['T'] = 1
    config['eps'] = eps
    return config