import logging
import os
import conf.config as config

file_handler = True
console_handler = True


def get_logger(logger_name=config.LOG.DEFAULT_LOGGER_NAME, log_path=config.LOG.DIR):

    # make sure that logger is not created multiple times
    if logger_name in logging.Logger.manager.loggerDict:
        return logging.getLogger(logger_name)

    return _setup_logger(logger_name, log_path)


def _setup_logger(logger_name, log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file = logger_name

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%b-%y-%H:%M:%S')

    # Create handlers
    if console_handler:
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)
        c_handler.setFormatter(log_formatter)
        logger.addHandler(c_handler)

    if file_handler:
        f_handler = logging.FileHandler("{0}/{1}.log".format(log_path, log_file))
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(log_formatter)
        logger.addHandler(f_handler)

    return logger
