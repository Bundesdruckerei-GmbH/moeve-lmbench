"""Configuration of the logging."""

log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "__main__": {"handlers": ["default"], "level": "DEBUG"},
        "lmbench": {"handlers": ["default"], "level": "DEBUG"},
    },
}
