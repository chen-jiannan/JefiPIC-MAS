"""Deprecated compatibility wrapper.

The TDIE implementation has been removed from this MAS branch.  Imports from
``EMsolver.tdie_integration`` are redirected to the MAS antenna interaction
classes so older notebooks fail less abruptly while users migrate.
"""

from .mas_integration import (
    MASAntennaInteraction,
    MASAntennaInteraction as PlasmaDeviceInteraction,
    SinusoidalCurrentExcitation,
    create_mas_antenna_simulation,
    create_plasma_device_simulation,
)

__all__ = [
    "MASAntennaInteraction",
    "PlasmaDeviceInteraction",
    "SinusoidalCurrentExcitation",
    "create_mas_antenna_simulation",
    "create_plasma_device_simulation",
]
