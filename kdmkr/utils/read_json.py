import json

__all__ = ['read_json']

def read_json(file_path):
    """Read entities and relations json."""
    return json.loads(open(file_path).read())
