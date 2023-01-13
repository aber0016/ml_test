import functools
import logging
import multiprocessing
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Callable

import torch
import csv
from fluidml.common import Resource
from fluidml.storage import LocalFileStore, TypeInfo
from rich.logging import RichHandler


logger = logging.getLogger(__name__)


class MyLocalFileStore(LocalFileStore):
    """
    save_fn: Callable                   # save function used to save the object
                                        to store
    load_fn: Callable                   # load function used to load the object
                                        from store
    extension: Optional[str] = None     # file extension the object is saved with
    is_binary: Optional[bool] = None    # read, write and append in binary mode
    open_fn: Optional[Callable] = None  # function used to open a file object
                                        (default is builtin open())
    needs_path: bool = False            # save and load fn operate on path and
                                        not on file like object
    """

    def __init__(self, base_dir: str, run_name: Optional[str] = None):
        super().__init__(base_dir=base_dir, run_name=run_name)

        self._type_registry["csv"] = TypeInfo(
            save_fn=self._save_csv,
            load_fn=self._load_csv,
            extension="csv",
            open_fn=self._save_csv,
            needs_path=True,
            is_binary=False,
        )

    @staticmethod
    def _save_csv(file_name: str, header: List[Any], data: List[Any]):
        with open(file_name, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write multiple rows
            writer.writerows(data)

    @staticmethod
    def _load_csv(file_: Any):
        csv_file = []
        with open(file_, newline="") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=" ", quotechar="|")
            for row in csv_reader:
                csv_file.append(row)

        return csv_file

    # @staticmethod
    # def _open_csv(file_: Any):
    #     pass


@dataclass
class TaskResource(Resource):
    device: Union[str, torch.device]


def configure_logging(level: Union[str, int] = "INFO", log_dir: Optional[str] = None):
    assert level in [
        "DEBUG",
        "INFO",
        "WARNING",
        "WARN",
        "ERROR",
        "FATAL",
        "CRITICAL",
        10,
        20,
        30,
        40,
        50,
    ]
    logger_ = logging.getLogger()
    formatter = logging.Formatter("%(processName)-13s%(message)s")
    stream_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_extra_lines=2,
        show_path=False,
        omit_repeated_times=False,
    )
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    if log_dir is not None:
        log_path = os.path.join(log_dir, f"{datetime.now()}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(processName)s - %(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger_.addHandler(stream_handler)
    logger_.setLevel(level)


def get_balanced_devices(
    count: Optional[int] = None,
    use_cuda: bool = True,
    cuda_ids: Optional[List[int]] = None,
) -> List[str]:
    count = count if count is not None else multiprocessing.cpu_count()
    if use_cuda and torch.cuda.is_available():
        if cuda_ids is not None:
            devices = [f"cuda:{id_}" for id_ in cuda_ids]
        else:
            devices = [f"cuda:{id_}" for id_ in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    factor = int(count / len(devices))
    remainder = count % len(devices)
    devices = devices * factor + devices[:remainder]
    return devices


def add_file_handler(
    log_dir: str,
    name: str = "logs",
    type_: str = "txt",
    level: Union[str, int] = "INFO",
):
    if level not in [
        "DEBUG",
        "INFO",
        "WARNING",
        "WARN",
        "ERROR",
        "FATAL",
        "CRITICAL",
        10,
        20,
        30,
        40,
        50,
    ]:
        raise ValueError(f'Logging level "{level}" is not supported.')

    log_path = os.path.join(log_dir, f"{name}.{type_}")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(processName)s - %(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger_ = logging.getLogger()
    logger_.addHandler(file_handler)


def remove_file_handler():
    logger_ = logging.getLogger()
    logger_.handlers = [
        h for h in logger_.handlers if not isinstance(h, logging.FileHandler)
    ]


def log_to_file(func):
    """Decorator to enable file logging for fluid ml tasks"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.log_to_file:
            run_dir = self.get_store_context()
            logger.info(f"Current run dir: {run_dir}")
            add_file_handler(run_dir)
            result = func(self, *args, **kwargs)
            remove_file_handler()
            return result
        return func(self, *args, **kwargs)

    return wrapper
