from enum import Enum


class ExitReason(Enum):
    NORMAL_EXIT = 0
    INVALID_INPUT = 1
    FILE_NOT_FOUND = 2
