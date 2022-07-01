import logging

str2log_level = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


def setup_logging_logger(log_file=None, log_level=logging.INFO):
    """
    Setup the logging logger.
    """
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=log_level,
                        handlers=[logging.StreamHandler()])
    if log_file is not None:
        logging.getLogger(__name__).addHandler(logging.FileHandler(log_file))
    return logging.getLogger(__name__)


def setup_loguru_logger(log_file=None, log_level=logging.INFO):
    """
    Setup logger.
    """
    from loguru import logger as loguru_logger
    if log_file is not None:
        loguru_logger.add(log_file, level=log_level)
    return loguru_logger


def setup_logger(log_file=None, log_level="INFO", backend="loguru"):
    """
    Setup logger.

    >>> logger = setup_logger(log_file="log.txt", log_level="INFO")
    >>> logger.info("Hello world!")
    """
    if isinstance(log_level, str):
        log_level = str2log_level[log_level.upper()]
    if backend == "loguru":
        logger = setup_loguru_logger(log_file, log_level)
    elif backend == "logging":
        logger = setup_logging_logger(log_file, log_level)
    else:
        raise ValueError("Unsupported backend: {}".format(backend))
    return logger
