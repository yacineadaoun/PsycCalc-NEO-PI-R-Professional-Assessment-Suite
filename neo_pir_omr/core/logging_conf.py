from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Optional


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "./logs"
    filename: str = "neo_pir_omr.log"
    max_bytes: int = 2_000_000
    backup_count: int = 3


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for k in ("scan_id", "file", "stage"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(cfg: LoggingConfig, logger_name: str = "neo_pir_omr") -> logging.Logger:
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
    logger.propagate = False

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    fmt = JsonFormatter()

    fh = RotatingFileHandler(
        os.path.join(cfg.log_dir, cfg.filename),
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
