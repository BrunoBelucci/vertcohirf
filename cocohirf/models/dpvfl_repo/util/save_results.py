import json
import os
from datetime import datetime
from json import JSONEncoder
import numpy

DIR = "./results"


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def save_result_to_json(result: dict, tag: str, experiment: str, dir: str = DIR):
    time = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S-%f")
    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(os.path.join(dir, tag + experiment)):
        os.mkdir(os.path.join(dir, tag + experiment))
    filename = os.path.join(dir, tag + experiment, experiment + time + ".json")
    json.dump(result, open(filename, "w"), indent=2, cls=NumpyArrayEncoder)
