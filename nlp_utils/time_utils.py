import time
import datetime


def get_current_datetime():
    """
    Output: `2022-06-23 18:54:28.321265`
    """
    return datetime.datetime.now()


def get_current_date(fmt='%Y%m%d'):
    return get_current_datetime_str(fmt)


def get_current_time(fmt='%H%M%S'):
    return get_current_datetime_str(fmt)


def get_current_datetime_str(fmt='%Y%m%d_%H%M%S'):
    """
    Output: `20220623_185428`
    """
    return datetime.datetime.now().strftime(fmt)