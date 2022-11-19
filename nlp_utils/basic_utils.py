import inspect


def get_var_name(var):
    loc = globals()
    for key in loc:
        if loc[key] is var:
            return key


def get_local_var_name(var):
    loc = locals()
    for key in loc:
        if loc[key] is var:
            return key


def get_class_name(obj):
    return obj.__class__.__name__


def get_function_name(func):
    return func.__name__


def get_current_class_name():
    return inspect.stack()[1][0].f_locals["self"].__class__.__name__


def get_current_function_name():
    """
    Example:
    >>> def dummy_func():
    >>>    print(get_current_function_name())
    >>> dummy_func()
    dummy_func
    """
    return inspect.stack()[1][3]


def get_current_file_name():
    return inspect.stack()[1][1].split("/")[-1]


def get_current_file_path():
    return inspect.stack()[1][1]
