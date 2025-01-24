from copy import deepcopy

def nested_update(original: dict, updates: dict) -> dict:
    original = deepcopy(original)
    def helper(original, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                helper(original[key], value)
            else:
                original[key] = value
    helper(original, updates)
    return original

def dict_to_str(d: dict) -> str:
    return '{' + ', '.join(f'"{k}": "{v}"' for k, v in d.items()) + '}'