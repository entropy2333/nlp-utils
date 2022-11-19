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


def memory_report():
    """
    Usage:
        python -c "from nlp_utils.train_utils import memory_report; memory_report()"

    Output:
        CPU Mem Usage: 25.1 GB / 67.3 GB
        GPU 0 Mem Usage: 481.0MB / 11448.0MB | Util  1%
        GPU 1 Mem Usage: 453.0MB / 11448.0MB | Util  1%
    """
    import psutil
    from humanize import naturalsize

    print(
        f"CPU Mem Usage: {naturalsize(psutil.virtual_memory().used)} / {naturalsize(psutil.virtual_memory().total)} |"
        f" Util {psutil.virtual_memory().percent:2.2f}%"
    )
    import GPUtil

    gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(gpus):
        print(f"GPU {i} Mem Usage: {gpu.memoryFree}MB / {gpu.memoryTotal}MB | Util {gpu.memoryUtil:2.2f}%")
