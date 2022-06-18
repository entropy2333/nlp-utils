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


def setup_logger(log_file=None, log_level="INFO"):
    """
    Setup logger.
    """
    log_level = str2log_level[log_level.upper()]
    logger = setup_logging_logger(log_file, log_level)
    return logger
