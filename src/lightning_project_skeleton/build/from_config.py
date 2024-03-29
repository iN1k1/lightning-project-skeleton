import importlib
import ast
from typing import Optional, Tuple, Union

from omegaconf import OmegaConf
import os.path as osp

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'


def __get_base_files(filename: str) -> list:
    """Get the base config file.

    Args:
        filename (str): The config file.

    Raises:
        TypeError: Name of config file.

    Returns:
        list: A list of base config.
    """
    file_format = osp.splitext(filename)[1]
    if file_format == '.py':
        with open(filename, encoding='utf-8') as f:
            parsed_codes = ast.parse(f.read()).body

            def is_base_line(c):
                return (isinstance(c, ast.Assign)
                        and isinstance(c.targets[0], ast.Name)
                        and c.targets[0].id == BASE_KEY)

            base_code = next((c for c in parsed_codes if is_base_line(c)),
                             None)
            if base_code is not None:
                base_code = ast.Expression(  # type: ignore
                    body=base_code.value)  # type: ignore
                base_files = eval(compile(base_code, '', mode='eval'))
            else:
                base_files = []
    elif file_format in ('.yml', '.yaml', '.json'):
        import mmengine
        cfg_dict = mmengine.load(filename)
        base_files = cfg_dict.get(BASE_KEY, [])
    else:
        raise FileNotFoundError(
            'The config type should be py, json, yaml or '
            f'yml, but got {file_format}')
    base_files = base_files if isinstance(base_files,
                                          list) else [base_files]
    return base_files

def __get_cfg_path(cfg_path: str,
                  filename: str) -> Tuple[str, Optional[str]]:
    """Get the config path from the current or external package.

    Args:
        cfg_path (str): Relative path of config.
        filename (str): The config file being parsed.

    Returns:
        Tuple[str, str or None]: Path and scope of config. If the config
        is not an external config, the scope will be `None`.
    """
    # Get local config path.
    cfg_dir = osp.dirname(filename)
    cfg_path = osp.join(cfg_dir, cfg_path)
    return cfg_path, None

def __merge_a_into_b(a: dict,
                    b: dict,
                    allow_list_keys: bool = False) -> dict:
    """merge dict ``a`` into dict ``b`` (non-inplace).

    Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
    in-place modifications.

    Args:
        a (dict): The source dict to be merged into ``b``.
        b (dict): The origin dict to be fetch keys from ``a``.
        allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
          are allowed in source ``a`` and will replace the element of the
          corresponding index in b if b is a list. Defaults to False.

    Returns:
        dict: The modified dict of ``b`` using ``a``.

    Examples:
        # Normally merge a into b.
        >>> Config._merge_a_into_b(
        ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
        {'obj': {'a': 2}}

        # Delete b first and merge a into b.
        >>> Config._merge_a_into_b(
        ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
        {'obj': {'a': 2}}

        # b is a list
        >>> Config._merge_a_into_b(
        ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
        [{'a': 2}, {'b': 2}]
    """
    b = b.copy()
    for k, v in a.items():
        if allow_list_keys and k.isdigit() and isinstance(b, list):
            k = int(k)
            if len(b) <= k:
                raise KeyError(f'Index {k} exceeds the length of list {b}')
            b[k] = __merge_a_into_b(v, b[k], allow_list_keys)
        elif isinstance(v, dict):
            if k in b and not v.pop(DELETE_KEY, False):
                allowed_types: Union[Tuple, type] = (
                    dict, list) if allow_list_keys else dict
                if not isinstance(b[k], allowed_types):
                    raise TypeError(
                        f'{k}={v} in child config cannot inherit from '
                        f'base because {k} is a dict in the child config '
                        f'but is of type {type(b[k])} in base config. '
                        f'You may set `{DELETE_KEY}=True` to ignore the '
                        f'base config.')
                b[k] = __merge_a_into_b(v, b[k], allow_list_keys)
            else:
                b[k] = v
        else:
            b[k] = v
    return b


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    print(config)
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_config_from_py_file(filename):
    filename = osp.abspath(osp.expanduser(filename))
    assert osp.exists(filename)
    fileExtname = osp.splitext(filename)[1]
    assert fileExtname in ['.py', '.json', '.yaml', '.yml']

    base_cfg_dict = {}
    for base_cfg_path in __get_base_files(filename):
        # base_cfg_path, scope = __get_cfg_path(base_cfg_path, filename)
        # cfg_dir = osp.dirname(filename)
        base_cfg_dir = osp.dirname(filename)
        base_cfg_path = osp.join(base_cfg_dir, base_cfg_path)

        _cfg_dict = load_config_from_py_file(filename=base_cfg_path)

        # cfg_text_list.append(_cfg_text)
        # env_variables.update(_env_variables)
        duplicate_keys = base_cfg_dict.keys() & _cfg_dict.keys()
        if len(duplicate_keys) > 0:
            raise KeyError(
                'Duplicate key is not allowed among bases. '
                f'Duplicate keys: {duplicate_keys}')

        # _dict_to_config_dict will do the following things:
        # 1. Recursively converts ``dict`` to :obj:`ConfigDict`.
        # 2. Set `_scope_` for the outer dict variable for the base
        # config.
        # 3. Set `scope` attribute for each base variable.
        # Different from `_scope_`， `scope` is not a key of base
        # dict, `scope` attribute will be parsed to key `_scope_`
        # by function `_parse_scope` only if the base variable is
        # accessed by the current config.
        # _cfg_dict = Config._dict_to_config_dict(_cfg_dict, scope)
        base_cfg_dict.update(_cfg_dict)

    if filename.endswith('.py'):

        with open(filename, encoding='utf-8') as f:
            parsed_codes = ast.parse(f.read())
        codeobj = compile(parsed_codes, filename, mode='exec')
        global_locals_var = {'base': dict()}
        ori_keys = set(global_locals_var.keys())
        eval(codeobj, global_locals_var, global_locals_var)
        cfg_dict = {
            key: value
            for key, value in global_locals_var.items()
            if (key not in ori_keys and not key.startswith('__'))
        }
    else:
        raise NotImplementedError('Handling only .py files')

    cfg_dict.pop(BASE_KEY, None)

    cfg_dict = __merge_a_into_b(cfg_dict, base_cfg_dict)
    cfg_dict = {
        k: v
        for k, v in cfg_dict.items() if not k.startswith('__')
    }
    return cfg_dict