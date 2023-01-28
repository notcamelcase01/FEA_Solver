"""
Defining necessary ENUMS instead of using numbers
"""
from enum import IntEnum


class ElementType(IntEnum):
    LINEAR = 2
    QUAD = 3


class RequiredParameters(IntEnum):
    STRESS = 1
    STRAIN = 2
