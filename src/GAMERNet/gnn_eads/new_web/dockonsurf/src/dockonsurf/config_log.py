"""Module for the configuration of how and what is recorded in the log file."""
import sys
import logging
import warnings


def log_exception(exc_type, exc_value, exc_tb):
    """Sets up the recording of exceptions on the log file

    @param exc_type: Type of exception
    @param exc_value: Value of the exception
    @param exc_tb:
    @return: None
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logger = logging.getLogger('DockOnSurf')
    logger.error("", exc_info=(exc_type, exc_value, exc_tb))


def log_warning(message, *args, **kwargs):
    """Sets up the recording of warnings on the log file

    @param message: Warning message.
    @param args: Additional arguments.
    @param kwargs: Additional keyword arguments.
    @return: None
    """
    logger = logging.getLogger('DockOnSurf')
    logger.warning(" ".join(f"{message}".split()))


def config_log(label):  # TODO Format log to break long lines (after column 80).
    """Configures the logger to record all calculation events on a log file.

    @param label: Label of the logger to be used.
    @return: The logger object.
    """
    logger = logging.getLogger(label)
    logger.setLevel(logging.INFO)

    log_handler = logging.FileHandler('dockonsurf.log', mode='w')
    log_handler.setLevel(logging.INFO)
    log_format = logging.Formatter(fmt='%(asctime)s-%(levelname)s: %(message)s',
                                   datefmt='%d-%b-%y %H:%M:%S')
    log_handler.setFormatter(log_format)

    logger.addHandler(log_handler)

    sys.excepthook = log_exception
    warnings.showwarning = log_warning

    return logger
