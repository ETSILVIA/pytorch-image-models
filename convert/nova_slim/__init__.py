import os
import time

# from .pruning import *
# from .compressor import Compressor, Pruner
# from .speedup import ModelSpeedup


import logging
import logging.config
import logging.handlers

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "log_dir": f"./logs/{''.join(['%02d' % x for x in time.localtime()[:6]])}/",
    "formatters": {
        "simple": {
            'format': '%(asctime)s [%(name)s] [%(levelname)s]- %(message)s'
        },
        'standard': {
            'format': '%(asctime)s [%(name)s] [%(levelname)s]- %(message)s'
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "debug": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "debug.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 6,
            "encoding": 'utf-8'
        },
        "info": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": "info.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 6,
            "encoding": 'utf-8'
        },
        "warn": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "WARN",
            "formatter": "simple",
            "filename": "warn.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 6,
            "encoding": 'utf-8'
        },
        "error": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "ERROR",
            "formatter": "simple",
            "filename": "error.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 6,
            "encoding": 'utf-8'
        }
    },

    "root": {
        'handlers': ['debug', "info", "warn", "error", "console"],
        'level': "DEBUG",
        'propagate': False
    }
}

os.makedirs(LOGGING_CONFIG["log_dir"], exist_ok=True)
for handler in LOGGING_CONFIG["handlers"].values():
    if "filename" in handler:
        handler["filename"] = os.path.join(LOGGING_CONFIG["log_dir"], handler["filename"])

logging.config.dictConfig(LOGGING_CONFIG)

