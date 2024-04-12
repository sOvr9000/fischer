


import jsonpickle # pip install jsonpickle
import json
import yaml


def object_to_str(obj: any, indent: int = 2):
	return json.dumps(json.loads(jsonpickle.encode(obj)), indent=indent)

def print_object(obj: any, indent: int = 2):
	print(object_to_str(obj, indent=indent))

def print_object_yaml(obj: any, indent: int = 2, file=None):
	print(yaml.dump(yaml.load(jsonpickle.encode(obj), yaml.FullLoader), indent=2), file=file)
