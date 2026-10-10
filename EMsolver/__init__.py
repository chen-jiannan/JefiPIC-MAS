"""EMsolver module for JefiPIC with MAS antenna coupling.

The CUDA EMsolver is imported lazily so CPU-only MAS preprocessing/tests do not
pay the CuPy/Numba startup cost or hang on machines without a CUDA runtime.
"""

from .region_distance import *
from .mas_preprocessor import (
    MASGeometry,
    assemble_impedance_matrix,
    build_mas_geometry,
    generate_cylindrical_antenna_points,
    generate_spherical_antenna_points,
    generate_stl_antenna_points,
    preprocess_mas_antenna,
)
from .mesh_guidance import auto_calculate_mesh, print_mesh_recommendation
from .mas_integration import (
    MASAntennaInteraction,
    PlasmaDeviceInteraction,
    SinusoidalCurrentExcitation,
    create_mas_antenna_simulation,
    create_plasma_device_simulation,
)


def __getattr__(name):
    if name == 'EMsolver':
        from .solver import EMsolver as _EMsolver
        return _EMsolver
    raise AttributeError(name)


__all__ = [
    'EMsolver',
    'MASGeometry',
    'assemble_impedance_matrix',
    'build_mas_geometry',
    'generate_cylindrical_antenna_points',
    'generate_spherical_antenna_points',
    'generate_stl_antenna_points',
    'preprocess_mas_antenna',
    'auto_calculate_mesh',
    'print_mesh_recommendation',
    'MASAntennaInteraction',
    'PlasmaDeviceInteraction',
    'SinusoidalCurrentExcitation',
    'create_mas_antenna_simulation',
    'create_plasma_device_simulation',
]
