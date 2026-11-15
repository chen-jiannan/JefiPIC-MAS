"""Compatibility preprocessing entry points for the MAS antenna workflow."""

from .mas_preprocessor import (
    MASGeometry,
    assemble_impedance_matrix,
    build_mas_geometry,
    generate_cylindrical_antenna_points,
    generate_spherical_antenna_points,
    preprocess_mas_antenna,
)

__all__ = [
    "MASGeometry",
    "assemble_impedance_matrix",
    "build_mas_geometry",
    "generate_cylindrical_antenna_points",
    "generate_spherical_antenna_points",
    "preprocess_mas_antenna",
]
