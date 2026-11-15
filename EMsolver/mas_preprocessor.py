"""
Method-of-Auxiliary-Sources (MAS) antenna geometry preprocessor.

This module replaces the old mesh/RWG/TDIE metal-device representation with a
point-cloud representation suitable for JefiPIC's integral field solver:

* ant_obs_points: physical PEC surface collocation/observation points.
* ant_normals: outward normals at the observation points.
* ant_src_points: auxiliary current-source points shifted inside the conductor.
* G / G_inv: a time-local MAS response operator used at each PIC step to enforce
  the tangential PEC boundary condition approximately.

The implementation deliberately keeps the preprocessing independent from the
particle grid.  The antenna geometry is therefore not voxelized; only the plasma
sources remain grid based.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings
import math

import numpy as np


_EPS = 1.0e-30


def _as_vec3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3,)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _normalize(value, name: str) -> np.ndarray:
    vec = _as_vec3(value, name)
    norm = float(np.linalg.norm(vec))
    if norm <= _EPS:
        raise ValueError(f"{name} must be non-zero")
    return vec / norm


def _orthonormal_basis(axis) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ez = _normalize(axis, "axis")
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, ez))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ex = helper - np.dot(helper, ez) * ez
    ex = ex / max(np.linalg.norm(ex), _EPS)
    ey = np.cross(ez, ex)
    ey = ey / max(np.linalg.norm(ey), _EPS)
    return ex, ey, ez


def build_local_tangent_frames(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return a deterministic orthonormal tangent pair for arbitrary normals.

    The construction is geometry-independent: for each normal the Cartesian
    axis least aligned with that normal is used as a helper.  The resulting
    pair spans the local tangent plane but does not assume cylindrical z/phi
    coordinates.
    """
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3).copy()
    nn = np.linalg.norm(n, axis=1)
    if np.any(nn <= _EPS):
        raise ValueError("All normals must be non-zero when building tangent frames")
    n /= nn[:, None]
    t1 = np.zeros_like(n)
    t2 = np.zeros_like(n)
    axes = np.eye(3, dtype=np.float64)
    for i, ni in enumerate(n):
        helper = axes[int(np.argmin(np.abs(ni)))]
        a = helper - np.dot(helper, ni) * ni
        a /= max(float(np.linalg.norm(a)), _EPS)
        b = np.cross(ni, a)
        b /= max(float(np.linalg.norm(b)), _EPS)
        t1[i] = a
        t2[i] = b
    return t1, t2


def _basis_matrix(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    n = len(t1)
    B = np.zeros((3*n, 2*n), dtype=np.float64)
    for i in range(n):
        B[3*i:3*i+3, 2*i] = t1[i]
        B[3*i:3*i+3, 2*i+1] = t2[i]
    return B


def _constraint_matrix(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    n = len(t1)
    C = np.zeros((2*n, 3*n), dtype=np.float64)
    for i in range(n):
        C[2*i, 3*i:3*i+3] = t1[i]
        C[2*i+1, 3*i:3*i+3] = t2[i]
    return C


@dataclass
class MASGeometry:
    """Container for the MAS antenna point cloud and its response matrix."""

    ant_obs_points: np.ndarray
    ant_src_points: np.ndarray
    ant_normals: np.ndarray
    source_weights: np.ndarray
    # Primary physical surface anchors are kept one-per-source for plotting,
    # surface-charge models, and source patch weights.  The Step-2G current
    # solve may use a denser independent collocation cloud.
    solve_obs_points: Optional[np.ndarray] = None
    solve_normals: Optional[np.ndarray] = None
    solve_weights: Optional[np.ndarray] = None
    source_tangent1: Optional[np.ndarray] = None
    source_tangent2: Optional[np.ndarray] = None
    solve_tangent1: Optional[np.ndarray] = None
    solve_tangent2: Optional[np.ndarray] = None
    surface_triangles: Optional[np.ndarray] = None
    surface_triangle_normals: Optional[np.ndarray] = None
    surface_triangle_areas: Optional[np.ndarray] = None
    G: Optional[np.ndarray] = None
    G_inv: Optional[np.ndarray] = None
    # Step-2B electric-dipole operator blocks.  These map surface-polarization
    # density P [C/m], surface-current density K=dP/dt [A/m], and dK/dt to
    # the physical electric field at MAS collocation points.  ``G_physical``
    # is the coefficient multiplying the current-node unknown K^n after the
    # trapezoidal P update and backward dK/dt are substituted.
    G_p: Optional[np.ndarray] = None
    G_k: Optional[np.ndarray] = None
    G_kdot: Optional[np.ndarray] = None
    G_physical: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.ant_obs_points = np.asarray(self.ant_obs_points, dtype=np.float64).reshape(-1, 3)
        self.ant_src_points = np.asarray(self.ant_src_points, dtype=np.float64).reshape(-1, 3)
        self.ant_normals = np.asarray(self.ant_normals, dtype=np.float64).reshape(-1, 3)
        self.source_weights = np.asarray(self.source_weights, dtype=np.float64).reshape(-1,)
        n = len(self.ant_src_points)
        if len(self.ant_obs_points) != n or len(self.ant_normals) != n or len(self.source_weights) != n:
            raise ValueError("Primary MAS source/anchor arrays must all have the same length")
        normal_norm = np.linalg.norm(self.ant_normals, axis=1)
        if np.any(normal_norm <= _EPS):
            raise ValueError("All MAS normals must be non-zero")
        self.ant_normals = self.ant_normals / normal_norm[:, None]

        if self.source_tangent1 is None or self.source_tangent2 is None:
            self.source_tangent1, self.source_tangent2 = build_local_tangent_frames(self.ant_normals)
        else:
            self.source_tangent1 = np.asarray(self.source_tangent1, dtype=np.float64).reshape(n, 3)
            self.source_tangent2 = np.asarray(self.source_tangent2, dtype=np.float64).reshape(n, 3)

        if self.solve_obs_points is None:
            self.solve_obs_points = self.ant_obs_points.copy()
        if self.solve_normals is None:
            self.solve_normals = self.ant_normals.copy()
        if self.solve_weights is None:
            self.solve_weights = self.source_weights.copy()
        self.set_solve_collocation(self.solve_obs_points, self.solve_normals, self.solve_weights)

        if self.surface_triangles is not None:
            self.surface_triangles = np.asarray(self.surface_triangles, dtype=np.float64).reshape(-1, 3, 3)
        if self.surface_triangle_normals is not None:
            self.surface_triangle_normals = np.asarray(self.surface_triangle_normals, dtype=np.float64).reshape(-1, 3)
        if self.surface_triangle_areas is not None:
            self.surface_triangle_areas = np.asarray(self.surface_triangle_areas, dtype=np.float64).reshape(-1)

    def set_solve_collocation(self, points, normals, weights=None):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        if len(points) != len(normals) or len(points) == 0:
            raise ValueError("solve collocation points/normals must be non-empty and have equal length")
        nn = np.linalg.norm(normals, axis=1)
        if np.any(nn <= _EPS):
            raise ValueError("solve collocation normals must be non-zero")
        normals = normals / nn[:, None]
        if weights is None:
            total_area = float(np.sum(self.source_weights))
            weights = np.full(len(points), total_area / len(points), dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(points):
            raise ValueError("solve collocation weights must match solve points")
        self.solve_obs_points = points
        self.solve_normals = normals
        self.solve_weights = weights
        self.solve_tangent1, self.solve_tangent2 = build_local_tangent_frames(normals)
        return self

    @property
    def num_points(self) -> int:
        """Number of auxiliary source patches (legacy name kept for API compatibility)."""
        return int(len(self.ant_src_points))

    @property
    def num_sources(self) -> int:
        return int(len(self.ant_src_points))

    @property
    def num_collocation_points(self) -> int:
        return int(len(self.solve_obs_points))

    def save_npz(self, path: str | Path) -> None:
        """Save the preprocessed geometry and matrices for reuse."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ant_obs_points=self.ant_obs_points,
            ant_src_points=self.ant_src_points,
            ant_normals=self.ant_normals,
            source_weights=self.source_weights,
            solve_obs_points=self.solve_obs_points,
            solve_normals=self.solve_normals,
            solve_weights=self.solve_weights,
            source_tangent1=self.source_tangent1,
            source_tangent2=self.source_tangent2,
            solve_tangent1=self.solve_tangent1,
            solve_tangent2=self.solve_tangent2,
            surface_triangles=np.asarray(self.surface_triangles) if self.surface_triangles is not None else np.zeros((0, 3, 3), dtype=np.float64),
            surface_triangle_normals=np.asarray(self.surface_triangle_normals) if self.surface_triangle_normals is not None else np.zeros((0, 3), dtype=np.float64),
            surface_triangle_areas=np.asarray(self.surface_triangle_areas) if self.surface_triangle_areas is not None else np.zeros((0,), dtype=np.float64),
            G=np.asarray(self.G) if self.G is not None else np.zeros((0, 0), dtype=np.float64),
            G_inv=np.asarray(self.G_inv) if self.G_inv is not None else np.zeros((0, 0), dtype=np.float64),
            G_p=np.asarray(self.G_p) if self.G_p is not None else np.zeros((0, 0), dtype=np.float64),
            G_k=np.asarray(self.G_k) if self.G_k is not None else np.zeros((0, 0), dtype=np.float64),
            G_kdot=np.asarray(self.G_kdot) if self.G_kdot is not None else np.zeros((0, 0), dtype=np.float64),
            G_physical=np.asarray(self.G_physical) if self.G_physical is not None else np.zeros((0, 0), dtype=np.float64),
            metadata=np.array([self.metadata], dtype=object),
        )

    @staticmethod
    def load_npz(path: str | Path) -> "MASGeometry":
        data = np.load(path, allow_pickle=True)
        metadata = {}
        if "metadata" in data and data["metadata"].size:
            metadata = dict(data["metadata"][0])
        G = data["G"] if "G" in data and data["G"].size else None
        G_inv = data["G_inv"] if "G_inv" in data and data["G_inv"].size else None
        G_p = data["G_p"] if "G_p" in data and data["G_p"].size else None
        G_k = data["G_k"] if "G_k" in data and data["G_k"].size else None
        G_kdot = data["G_kdot"] if "G_kdot" in data and data["G_kdot"].size else None
        G_physical = data["G_physical"] if "G_physical" in data and data["G_physical"].size else None
        return MASGeometry(
            ant_obs_points=data["ant_obs_points"],
            ant_src_points=data["ant_src_points"],
            ant_normals=data["ant_normals"],
            source_weights=data["source_weights"],
            solve_obs_points=data["solve_obs_points"] if "solve_obs_points" in data and data["solve_obs_points"].size else None,
            solve_normals=data["solve_normals"] if "solve_normals" in data and data["solve_normals"].size else None,
            solve_weights=data["solve_weights"] if "solve_weights" in data and data["solve_weights"].size else None,
            source_tangent1=data["source_tangent1"] if "source_tangent1" in data and data["source_tangent1"].size else None,
            source_tangent2=data["source_tangent2"] if "source_tangent2" in data and data["source_tangent2"].size else None,
            solve_tangent1=data["solve_tangent1"] if "solve_tangent1" in data and data["solve_tangent1"].size else None,
            solve_tangent2=data["solve_tangent2"] if "solve_tangent2" in data and data["solve_tangent2"].size else None,
            surface_triangles=data["surface_triangles"] if "surface_triangles" in data and data["surface_triangles"].size else None,
            surface_triangle_normals=data["surface_triangle_normals"] if "surface_triangle_normals" in data and data["surface_triangle_normals"].size else None,
            surface_triangle_areas=data["surface_triangle_areas"] if "surface_triangle_areas" in data and data["surface_triangle_areas"].size else None,
            G=G,
            G_inv=G_inv,
            G_p=G_p,
            G_k=G_k,
            G_kdot=G_kdot,
            G_physical=G_physical,
            metadata=metadata,
        )


def generate_cylindrical_antenna_points(
    radius: float,
    length: float,
    center=(0.0, 0.0, 0.0),
    axis=(0.0, 0.0, 1.0),
    n_length: int = 32,
    n_phi: int = 12,
    source_offset: Optional[float] = None,
) -> MASGeometry:
    """Generate a wire/cylindrical antenna as MAS observation/source points.

    The physical observation points lie on the conductor surface. Auxiliary
    sources are shifted inward along the local normal by ``source_offset``.
    """
    radius = float(radius)
    length = float(length)
    n_length = int(n_length)
    n_phi = int(n_phi)
    if radius <= 0.0 or length <= 0.0:
        raise ValueError("radius and length must be positive")
    if n_length < 2 or n_phi < 3:
        raise ValueError("Use n_length >= 2 and n_phi >= 3 for a cylindrical MAS antenna")
    if source_offset is None:
        source_offset = 0.35 * radius
    source_offset = float(source_offset)
    if not (0.0 < source_offset < radius):
        raise ValueError("source_offset must be inside the conductor: 0 < source_offset < radius")

    center = _as_vec3(center, "center")
    ex, ey, ez = _orthonormal_basis(axis)

    z_values = (np.arange(n_length, dtype=np.float64) + 0.5) / n_length - 0.5
    z_values *= length
    phi_values = 2.0 * np.pi * (np.arange(n_phi, dtype=np.float64) + 0.5) / n_phi

    obs = []
    src = []
    normals = []
    weights = []
    patch_area = (length / n_length) * (2.0 * np.pi * radius / n_phi)
    for z in z_values:
        axial = center + z * ez
        for phi in phi_values:
            radial_dir = np.cos(phi) * ex + np.sin(phi) * ey
            obs_point = axial + radius * radial_dir
            src_point = axial + (radius - source_offset) * radial_dir
            obs.append(obs_point)
            src.append(src_point)
            normals.append(radial_dir)
            weights.append(patch_area)

    return MASGeometry(
        ant_obs_points=np.asarray(obs, dtype=np.float64),
        ant_src_points=np.asarray(src, dtype=np.float64),
        ant_normals=np.asarray(normals, dtype=np.float64),
        source_weights=np.asarray(weights, dtype=np.float64),
        metadata={
            "kind": "cylinder",
            "radius": radius,
            "length": length,
            "center": center.tolist(),
            "axis": _normalize(axis, "axis").tolist(),
            "n_length": n_length,
            "n_phi": n_phi,
            "source_offset": source_offset,
        },
    )



def _triangle_geometry(triangles: np.ndarray):
    """Return triangle centroids, outward unit normals, and areas."""
    tri = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cr = np.cross(e1, e2)
    nn = np.linalg.norm(cr, axis=1)
    area = 0.5 * nn
    normal = cr / np.maximum(nn[:, None], _EPS)
    cent = np.mean(tri, axis=1)
    return cent, normal, area


def _fps_indices(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Deterministic farthest-point sampling used by the validated Step-2M chain."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    N = len(P)
    n = min(max(int(n), 0), N)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    start = int(seed) % N
    chosen = np.empty(n, dtype=np.int64)
    chosen[0] = start
    d2 = np.sum((P - P[start])**2, axis=1)
    d2[start] = -1.0
    for k in range(1, n):
        j = int(np.argmax(d2))
        chosen[k] = j
        nd2 = np.sum((P - P[j])**2, axis=1)
        d2 = np.minimum(d2, nd2)
        d2[chosen[:k+1]] = -1.0
    return chosen


def _nearest_partition_weights(candidate_points: np.ndarray,
                               candidate_weights: np.ndarray,
                               selected_points: np.ndarray) -> np.ndarray:
    """Voronoi-like area accumulation onto selected FPS points."""
    from scipy.spatial import cKDTree
    candidate_points = np.asarray(candidate_points, dtype=np.float64).reshape(-1, 3)
    candidate_weights = np.asarray(candidate_weights, dtype=np.float64).reshape(-1)
    selected_points = np.asarray(selected_points, dtype=np.float64).reshape(-1, 3)
    _, labels = cKDTree(selected_points).query(candidate_points, k=1)
    return np.bincount(
        labels, weights=candidate_weights, minlength=len(selected_points)
    ).astype(np.float64)


def _project_to_capsule(points: np.ndarray, radius: float,
                        physical_radius: float, length: float):
    """Project onto a capsule around the z centerline [Rphysical, L-Rphysical]."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    z0, z1 = float(physical_radius), float(length) - float(physical_radius)
    qz = np.clip(P[:, 2], z0, z1)
    Q = np.column_stack((np.zeros(len(P)), np.zeros(len(P)), qz))
    V = P - Q
    nv = np.linalg.norm(V, axis=1)
    N = V / np.maximum(nv[:, None], _EPS)
    return Q + float(radius) * N, N


def _orient_capsule_triangles_outward(triangles: np.ndarray,
                                      physical_radius: float,
                                      length: float) -> np.ndarray:
    T = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3).copy()
    C, N, _ = _triangle_geometry(T)
    _, A = _project_to_capsule(C, float(physical_radius), float(physical_radius), float(length))
    flip = np.sum(N*A, axis=1) < 0.0
    if np.any(flip):
        tmp = T[flip, 1].copy()
        T[flip, 1] = T[flip, 2]
        T[flip, 2] = tmp
    return T


def _generate_capsule_triangles(radius: float, physical_radius: float, length: float,
                                n_cyl: int, n_phi: int, n_hemi: int) -> np.ndarray:
    """Triangulated smooth capsule matching the validated Step-2M construction."""
    radius = float(radius)
    physical_radius = float(physical_radius)
    length = float(length)
    n_cyl = int(n_cyl); n_phi = int(n_phi); n_hemi = int(n_hemi)
    if radius <= 0.0 or physical_radius <= 0.0:
        raise ValueError("capsule radii must be positive")
    if length <= 2.0*physical_radius:
        raise ValueError("arm length must exceed 2*physical_radius")
    if n_cyl < 2 or n_phi < 6 or n_hemi < 2:
        raise ValueError("Use n_cyl>=2, n_phi>=6, n_hemi>=2")

    ph = 2.0*np.pi*np.arange(n_phi)/n_phi
    rings = []
    for th in np.linspace(-0.5*np.pi, 0.0, n_hemi+1)[1:]:
        rings.append((
            float(physical_radius + radius*np.sin(th)),
            float(radius*np.cos(th)),
        ))
    z0, z1 = physical_radius, length-physical_radius
    for zz in np.linspace(z0, z1, n_cyl+1)[1:]:
        rings.append((float(zz), float(radius)))
    for th in np.linspace(0.0, 0.5*np.pi, n_hemi+1)[1:-1]:
        rings.append((
            float((length-physical_radius) + radius*np.sin(th)),
            float(radius*np.cos(th)),
        ))

    verts = [
        np.column_stack((rr*np.cos(ph), rr*np.sin(ph), np.full(n_phi, zz)))
        for zz, rr in rings
    ]
    bottom = np.array([0.0, 0.0, physical_radius-radius])
    top = np.array([0.0, 0.0, length-physical_radius+radius])
    tris = []
    first = verts[0]
    for j in range(n_phi):
        k = (j+1) % n_phi
        tris.append(np.asarray([bottom, first[j], first[k]], dtype=np.float64))
    for a, b in zip(verts[:-1], verts[1:]):
        for j in range(n_phi):
            k = (j+1) % n_phi
            tris.append(np.asarray([a[j], b[j], b[k]], dtype=np.float64))
            tris.append(np.asarray([a[j], b[k], a[k]], dtype=np.float64))
    last = verts[-1]
    for j in range(n_phi):
        k = (j+1) % n_phi
        tris.append(np.asarray([last[j], top, last[k]], dtype=np.float64))
    return _orient_capsule_triangles_outward(
        np.asarray(tris, dtype=np.float64), physical_radius, length
    )


def _shift_z(array: np.ndarray, dz: float) -> np.ndarray:
    out = np.asarray(array, dtype=np.float64).copy()
    out[..., 2] += float(dz)
    return out


def generate_rounded_two_arm_dipole_points(
    radius: float,
    total_length: float,
    feed_gap: float,
    depth_frac: float = 0.80,
    sources_per_arm: int = 900,
    collocation_factor: float = 2.0,
    n_phi: int = 72,
    n_cyl: int = 96,
    n_hemi: int = 24,
    source_seed_base: int = 6100,
    collocation_seed_base: int = 5100,
) -> MASGeometry:
    """Validated smooth two-arm rounded dipole production geometry.

    This is the restricted Step-2M formulation promoted into production for
    Step 3-A regression.  It intentionally targets smooth rounded arms only;
    it is not a generic sharp-STL construction.

    Source placement:
      * coherent inner capsule at d/R = ``depth_frac``;
      * deterministic global farthest-point sampling per arm;
      * source patch weights from inner-surface Voronoi areas rescaled to the
        corresponding physical arm area.

    Collocation:
      * independent physical-surface FPS cloud;
      * area-partition weights;
      * oversampling controlled by ``collocation_factor``.
    """
    R = float(radius)
    total = float(total_length)
    gap = float(feed_gap)
    frac = float(depth_frac)
    nsrc = int(sources_per_arm)
    if R <= 0.0 or total <= 0.0 or gap <= 0.0:
        raise ValueError("radius, total_length, and feed_gap must be positive")
    if not (0.0 < frac < 1.0):
        raise ValueError("depth_frac must lie in (0,1)")
    Larm = 0.5*(total-gap)
    if Larm <= 2.0*R:
        raise ValueError("Each arm must be longer than 2*radius")
    if nsrc < 4:
        raise ValueError("sources_per_arm must be at least 4")
    ncoll = max(nsrc+1, int(np.ceil(float(collocation_factor)*nsrc)))
    d = frac*R
    Rs = (1.0-frac)*R
    z0s = (-0.5*total, 0.5*gap)

    physical_local = _generate_capsule_triangles(R, R, Larm, n_cyl, n_phi, n_hemi)
    inner_local = _generate_capsule_triangles(Rs, R, Larm, n_cyl, n_phi, n_hemi)
    Cphys0, _, Aphys = _triangle_geometry(physical_local)
    Cphys, Nphys = _project_to_capsule(Cphys0, R, R, Larm)
    Cinner0, _, Ainner = _triangle_geometry(inner_local)
    Cinner, Ninner = _project_to_capsule(Cinner0, Rs, R, Larm)
    area_arm = float(np.sum(Aphys))

    anchors=[]; sources=[]; normals=[]; sweights=[]
    solve_points=[]; solve_normals=[]; solve_weights=[]
    all_physical_tri=[]
    for arm, zshift in enumerate(z0s):
        # Frozen MB1 samples AFTER translating each arm into global coordinates.
        # On this highly symmetric capsule, FPS has many nearly tied farthest
        # candidates.  Translation is exact mathematically, but floating-point
        # roundoff changes argmax tie-breaking, so local-FPS-then-shift does NOT
        # reproduce the validated MB1 cloud.
        Cinner_global = _shift_z(Cinner, zshift)
        Cphys_global = _shift_z(Cphys, zshift)

        # Source sampling seeds exactly match MB1.
        sidx = _fps_indices(Cinner_global, nsrc, seed=source_seed_base + arm)
        SP = Cinner_global[sidx].copy()
        SN = Ninner[sidx].copy()
        SW = _nearest_partition_weights(Cinner_global, Ainner, SP)
        SW *= area_arm / max(float(np.sum(SW)), _EPS)

        # Physical anchor at the corresponding capsule parameter location.
        AP_local, AN = _project_to_capsule(Cinner[sidx], R, R, Larm)
        AP = _shift_z(AP_local, zshift)

        # Independent collocation seeds exactly match MB1.
        cidx = _fps_indices(Cphys_global, ncoll, seed=collocation_seed_base + arm)
        CP = Cphys_global[cidx].copy()
        CN = Nphys[cidx].copy()
        CW = _nearest_partition_weights(Cphys_global, Aphys, CP)

        anchors.append(AP); sources.append(SP); normals.append(SN); sweights.append(SW)
        solve_points.append(CP); solve_normals.append(CN); solve_weights.append(CW)
        all_physical_tri.append(_shift_z(physical_local, zshift))

    g = MASGeometry(
        ant_obs_points=np.concatenate(anchors),
        ant_src_points=np.concatenate(sources),
        ant_normals=np.concatenate(normals),
        source_weights=np.concatenate(sweights),
        solve_obs_points=np.concatenate(solve_points),
        solve_normals=np.concatenate(solve_normals),
        solve_weights=np.concatenate(solve_weights),
        surface_triangles=np.concatenate(all_physical_tri),
        metadata={
            "kind": "rounded_two_arm_dipole",
            "validated_scope": "smooth_rounded_two_arm_feed_gap",
            "radius": R,
            "total_length": total,
            "feed_gap": gap,
            "arm_length": Larm,
            "depth_frac": frac,
            "source_offset": d,
            "inner_radius": Rs,
            "sources_per_arm": nsrc,
            "collocation_points_per_arm": ncoll,
            "collocation_factor": float(collocation_factor),
            "n_phi": int(n_phi),
            "n_cyl": int(n_cyl),
            "n_hemi": int(n_hemi),
            "source_sampling": "global_fps_per_arm",
            "collocation_sampling": "independent_global_fps_per_arm",
            "source_weighting": "inner_voronoi_rescaled_to_physical_arm_area",
        },
    )
    # Source tangent frames must use the coherent inner-surface normals.
    g.source_tangent1, g.source_tangent2 = build_local_tangent_frames(
        np.concatenate(normals)
    )
    return g



def _shift_x(array: np.ndarray, dx: float) -> np.ndarray:
    out = np.asarray(array, dtype=np.float64).copy()
    out[..., 0] += float(dx)
    return out


def generate_parallel_two_rounded_dipoles_points(
    radius: float,
    total_length: float,
    feed_gap: float,
    antenna_separation: float,
    depth_frac: float = 0.80,
    sources_per_arm: int = 600,
    collocation_factor: float = 2.0,
    n_phi: int = 72,
    n_cyl: int = 96,
    n_hemi: int = 24,
    dual_seed_offset: int = 20,
) -> MASGeometry:
    """Frozen Step-2 MD2 two-parallel-dipole geometry for production use.

    This production construction is deliberately restricted to the validated
    MD2 configuration family: two parallel smooth rounded two-arm dipoles.
    The per-arm FPS seeds exactly reproduce the frozen MD2 ``case_name='dual'``
    sampling convention.

    No density qualification is performed here.  Formal Step 3-A3 uses the
    Step-2 MD1-qualified density of 600 sources/arm.
    """
    R = float(radius)
    total = float(total_length)
    gap = float(feed_gap)
    sep = float(antenna_separation)
    frac = float(depth_frac)
    nsrc = int(sources_per_arm)
    if R <= 0.0 or total <= 0.0 or gap <= 0.0 or sep <= 2.0*R:
        raise ValueError("Require R>0, total>0, gap>0, and separation>2R")
    if not (0.0 < frac < 1.0):
        raise ValueError("depth_frac must lie in (0,1)")
    Larm = 0.5*(total-gap)
    if Larm <= 2.0*R:
        raise ValueError("Each arm must be longer than 2*radius")
    if nsrc < 4:
        raise ValueError("sources_per_arm must be at least 4")

    ncoll = max(nsrc+1, int(np.ceil(float(collocation_factor)*nsrc)))
    d = frac*R
    Rs = (1.0-frac)*R
    x_centers = (-0.5*sep, 0.5*sep)
    z0s = (-0.5*total, 0.5*gap)

    physical_local = _generate_capsule_triangles(R, R, Larm, n_cyl, n_phi, n_hemi)
    inner_local = _generate_capsule_triangles(Rs, R, Larm, n_cyl, n_phi, n_hemi)

    Cphys0, _, Aphys = _triangle_geometry(physical_local)
    Cphys, Nphys = _project_to_capsule(Cphys0, R, R, Larm)
    Cinner0, _, Ainner = _triangle_geometry(inner_local)
    Cinner, Ninner = _project_to_capsule(Cinner0, Rs, R, Larm)
    area_arm = float(np.sum(Aphys))

    anchors=[]; sources=[]; normals=[]; sweights=[]
    solve_points=[]; solve_normals=[]; solve_weights=[]
    physical_triangles=[]; source_ant=[]; source_side=[]

    # Ordering exactly matches frozen MD2: ant0 lower, ant0 upper,
    # ant1 lower, ant1 upper.
    for ant, x0 in enumerate(x_centers):
        for side, z0 in enumerate(z0s):
            # Frozen MD2 make_shifted_arm performs z translation first, then x.
            Cphys_global = _shift_x(_shift_z(Cphys, z0), x0)
            Cinner_global = _shift_x(_shift_z(Cinner, z0), x0)

            cseed = 5100 + 100*ant + side + int(dual_seed_offset)
            sseed = 6100 + 100*ant + side + int(dual_seed_offset)

            cidx = _fps_indices(Cphys_global, ncoll, seed=cseed)
            CP = Cphys_global[cidx].copy()
            CN = Nphys[cidx].copy()
            CW = _nearest_partition_weights(Cphys_global, Aphys, CP)

            sidx = _fps_indices(Cinner_global, nsrc, seed=sseed)
            SP = Cinner_global[sidx].copy()
            SN = Ninner[sidx].copy()
            SW = _nearest_partition_weights(Cinner_global, Ainner, SP)
            SW *= area_arm / max(float(np.sum(SW)), _EPS)

            AP_local, _ = _project_to_capsule(Cinner[sidx], R, R, Larm)
            AP = _shift_x(_shift_z(AP_local, z0), x0)

            anchors.append(AP); sources.append(SP); normals.append(SN); sweights.append(SW)
            solve_points.append(CP); solve_normals.append(CN); solve_weights.append(CW)
            physical_triangles.append(_shift_x(_shift_z(physical_local, z0), x0))
            source_ant.append(np.full(nsrc, ant, dtype=np.int32))
            source_side.append(np.full(nsrc, side, dtype=np.int32))

    source_ant_arr = np.concatenate(source_ant)
    source_side_arr = np.concatenate(source_side)
    g = MASGeometry(
        ant_obs_points=np.concatenate(anchors),
        ant_src_points=np.concatenate(sources),
        ant_normals=np.concatenate(normals),
        source_weights=np.concatenate(sweights),
        solve_obs_points=np.concatenate(solve_points),
        solve_normals=np.concatenate(solve_normals),
        solve_weights=np.concatenate(solve_weights),
        surface_triangles=np.concatenate(physical_triangles),
        metadata={
            "kind": "parallel_two_rounded_dipoles",
            "validated_scope": "MD2_parallel_two_smooth_rounded_dipoles",
            "radius": R,
            "total_length": total,
            "feed_gap": gap,
            "arm_length": Larm,
            "antenna_separation": sep,
            "depth_frac": frac,
            "source_offset": d,
            "inner_radius": Rs,
            "sources_per_arm": nsrc,
            "collocation_points_per_arm": ncoll,
            "collocation_factor": float(collocation_factor),
            "n_phi": int(n_phi),
            "n_cyl": int(n_cyl),
            "n_hemi": int(n_hemi),
            "dual_seed_offset": int(dual_seed_offset),
            "source_sampling": "MD2_global_fps_per_arm",
            "collocation_sampling": "MD2_independent_global_fps_per_arm",
            "source_weighting": "inner_voronoi_rescaled_to_physical_arm_area",
            "source_antenna_ids": source_ant_arr.tolist(),
            "source_arm_sides": source_side_arr.tolist(),
            "density_qualification": "Step2_MD1_600_sources_per_arm_formal",
            "mutual_response_proxy_note": "Coefficient L2 norm ratio B/A; not S21 or calibrated port current.",
        },
    )
    g.source_tangent1, g.source_tangent2 = build_local_tangent_frames(
        np.concatenate(normals)
    )
    return g

def _bent_tube_frame(theta: float):
    """Planar x-z elbow frame used by the frozen Step-2 MC geometry."""
    t = np.array([math.sin(theta), 0.0, math.cos(theta)], dtype=np.float64)
    n = np.array([math.cos(theta), 0.0, -math.sin(theta)], dtype=np.float64)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return t, n, b


def _bent_centerline_geometry(Rphysical: float, Larm: float, gap: float,
                              Rbend: float, bend_angle: float):
    centerline_len = float(Larm - 2.0*Rphysical)
    arc_len = float(bend_angle*Rbend)
    straight_each = 0.5*(centerline_len - arc_len)
    if straight_each <= 1.25*Rphysical:
        raise ValueError(
            "Bend radius/angle leave insufficient straight segment length."
        )
    p0 = np.array([0.0, 0.0, 0.5*gap + Rphysical], dtype=np.float64)
    p_arc0 = p0 + np.array([0.0, 0.0, straight_each], dtype=np.float64)
    center = p_arc0 + np.array([Rbend, 0.0, 0.0], dtype=np.float64)
    p_arc1 = p_arc0 + np.array([
        Rbend*(1.0-math.cos(bend_angle)), 0.0, Rbend*math.sin(bend_angle)
    ], dtype=np.float64)
    t_end = np.array([math.sin(bend_angle), 0.0, math.cos(bend_angle)], dtype=np.float64)
    p1 = p_arc1 + straight_each*t_end
    return {
        "Rphysical": float(Rphysical),
        "Larm": float(Larm),
        "gap": float(gap),
        "Rbend": float(Rbend),
        "bend_angle": float(bend_angle),
        "centerline_len": centerline_len,
        "arc_len": arc_len,
        "straight_pre": straight_each,
        "straight_post": straight_each,
        "p0": p0,
        "p_arc0": p_arc0,
        "bend_center": center,
        "p_arc1": p_arc1,
        "p1": p1,
        "s_arc0": straight_each,
        "s_arc1": straight_each + arc_len,
        "s_end": centerline_len,
    }


def _bent_centerline_sample(geom, s: float):
    s = float(np.clip(s, 0.0, geom["s_end"]))
    if s <= geom["s_arc0"]:
        c = geom["p0"] + np.array([0.0, 0.0, s], dtype=np.float64)
        t, n, b = _bent_tube_frame(0.0)
        kind = "tube_pre"; theta = 0.0
    elif s <= geom["s_arc1"]:
        u = s - geom["s_arc0"]
        theta = min(geom["bend_angle"], u/geom["Rbend"])
        c = geom["p_arc0"] + np.array([
            geom["Rbend"]*(1.0-math.cos(theta)),
            0.0,
            geom["Rbend"]*math.sin(theta),
        ], dtype=np.float64)
        t, n, b = _bent_tube_frame(theta)
        kind = "tube_arc"
    else:
        u = s - geom["s_arc1"]
        t, n, b = _bent_tube_frame(geom["bend_angle"])
        c = geom["p_arc1"] + u*t
        kind = "tube_post"; theta = geom["bend_angle"]
    return c, t, n, b, kind, float(theta)


def _generate_bent_capsule_triangles(radius: float, Rphysical: float, Larm: float,
                                     gap: float, Rbend: float, bend_angle: float,
                                     n_pre: int, n_arc: int, n_post: int,
                                     n_phi: int, n_hemi: int):
    """Exact production port of the frozen Step-2 MC C1 bent capsule."""
    geom = _bent_centerline_geometry(Rphysical, Larm, gap, Rbend, bend_angle)
    ph = 2.0*np.pi*np.arange(int(n_phi))/int(n_phi)

    spre = np.linspace(0.0, geom["s_arc0"], int(n_pre)+1)
    sarc = np.linspace(geom["s_arc0"], geom["s_arc1"], int(n_arc)+1)[1:]
    spost = np.linspace(geom["s_arc1"], geom["s_end"], int(n_post)+1)[1:]
    svals = np.concatenate((spre, sarc, spost))

    rings=[]; ring_data=[]
    for s in svals:
        c,t,n,b,kind,theta = _bent_centerline_sample(geom, s)
        ring = c[None,:] + radius*(
            np.cos(ph)[:,None]*n[None,:] + np.sin(ph)[:,None]*b[None,:]
        )
        rings.append(ring)
        ring_data.append((float(s), kind, float(theta)))

    tris=[]; meta=[]
    def add(tri, **kw):
        tris.append(np.asarray(tri, dtype=np.float64)); meta.append(dict(kw))

    for ir in range(len(rings)-1):
        a, br = rings[ir], rings[ir+1]
        s0, _, _ = ring_data[ir]; s1, _, _ = ring_data[ir+1]
        sm = 0.5*(s0+s1)
        _,_,_,_,kindm,thm = _bent_centerline_sample(geom, sm)
        for j in range(int(n_phi)):
            k=(j+1)%int(n_phi)
            phim=(2.0*np.pi*(j+0.5)/int(n_phi))%(2.0*np.pi)
            common=dict(kind=kindm,s=sm,theta=thm,phi=phim,beta=np.nan)
            add([a[j],br[j],br[k]],**common)
            add([a[j],br[k],a[k]],**common)

    c0,t0,n0,b0,_,_ = _bent_centerline_sample(geom,0.0)
    pole0=c0-radius*t0
    betas=np.linspace(0.0,0.5*np.pi,int(n_hemi)+1)[1:]
    cap=[]
    for beta in betas:
        cap.append(c0[None,:]+radius*(
            -math.cos(beta)*t0[None,:]
            +math.sin(beta)*(np.cos(ph)[:,None]*n0[None,:]+np.sin(ph)[:,None]*b0[None,:])
        ))
    first=cap[0]; beta_mid=0.5*betas[0]
    for j in range(int(n_phi)):
        k=(j+1)%int(n_phi); phim=(2*np.pi*(j+0.5)/int(n_phi))%(2*np.pi)
        add([pole0,first[k],first[j]],kind="feed_cap",s=0.0,theta=0.0,phi=phim,beta=beta_mid)
    for ir in range(len(cap)-1):
        a,br=cap[ir],cap[ir+1]; beta_mid=0.5*(betas[ir]+betas[ir+1])
        for j in range(int(n_phi)):
            k=(j+1)%int(n_phi); phim=(2*np.pi*(j+0.5)/int(n_phi))%(2*np.pi)
            common=dict(kind="feed_cap",s=0.0,theta=0.0,phi=phim,beta=beta_mid)
            add([a[j],br[k],br[j]],**common); add([a[j],a[k],br[k]],**common)

    c1,t1,n1,b1,_,_ = _bent_centerline_sample(geom,geom["s_end"])
    pole1=c1+radius*t1
    cap=[]
    for beta in betas[::-1]:
        cap.append(c1[None,:]+radius*(
            math.cos(beta)*t1[None,:]
            +math.sin(beta)*(np.cos(ph)[:,None]*n1[None,:]+np.sin(ph)[:,None]*b1[None,:])
        ))
    bdesc=betas[::-1]
    for ir in range(len(cap)-1):
        a,br=cap[ir],cap[ir+1]; beta_mid=0.5*(bdesc[ir]+bdesc[ir+1])
        for j in range(int(n_phi)):
            k=(j+1)%int(n_phi); phim=(2*np.pi*(j+0.5)/int(n_phi))%(2*np.pi)
            common=dict(kind="outer_cap",s=geom["s_end"],theta=geom["bend_angle"],
                        phi=phim,beta=beta_mid)
            add([a[j],br[j],br[k]],**common); add([a[j],br[k],a[k]],**common)
    last=cap[-1]; beta_mid=0.5*bdesc[-1]
    for j in range(int(n_phi)):
        k=(j+1)%int(n_phi); phim=(2*np.pi*(j+0.5)/int(n_phi))%(2*np.pi)
        # Match the frozen Step-2 MC vertex order exactly.  Although these
        # three vertices define the same triangle geometrically, changing their
        # order changes floating-point centroid summation at the outer pole.
        # On the highly symmetric inner bent surface that is sufficient to
        # alter FPS tie-breaking and therefore the entire source sequence.
        add([last[j],last[k],pole1],kind="outer_cap",s=geom["s_end"],
            theta=geom["bend_angle"],phi=phim,beta=beta_mid)

    return np.asarray(tris,dtype=np.float64), meta, geom


def _project_bent_by_meta(points: np.ndarray, meta, geom, radius: float):
    P=np.asarray(points,dtype=np.float64).reshape(-1,3)
    Q=np.empty_like(P); N=np.empty_like(P)
    for i,(p,m) in enumerate(zip(P,meta)):
        kind=m["kind"]
        if kind=="feed_cap":
            c=geom["p0"]; v=p-c
        elif kind=="outer_cap":
            c=geom["p1"]; v=p-c
        else:
            cands=[]
            u=float(np.clip(p[2]-geom["p0"][2],0.0,geom["straight_pre"]))
            cands.append(geom["p0"]+np.array([0.0,0.0,u]))
            qx=float(p[0]-geom["bend_center"][0]); qz=float(p[2]-geom["bend_center"][2])
            th=float(np.clip(math.atan2(qz,-qx),0.0,geom["bend_angle"]))
            cands.append(geom["p_arc0"]+np.array([
                geom["Rbend"]*(1.0-math.cos(th)),0.0,geom["Rbend"]*math.sin(th)
            ]))
            tend=np.array([np.sin(geom["bend_angle"]),0.0,np.cos(geom["bend_angle"])])
            u=float(np.clip(np.dot(p-geom["p_arc1"],tend),0.0,geom["straight_post"]))
            cands.append(geom["p_arc1"]+u*tend)
            dd=[float(np.dot(p-c,p-c)) for c in cands]
            c=cands[int(np.argmin(dd))]
            v=p-c
        nv=float(np.linalg.norm(v)); n=v/max(nv,_EPS)
        Q[i]=c+radius*n; N[i]=n
    return Q,N


def _bent_candidates(triangles: np.ndarray, meta, geom, radius: float):
    C0,_,A=_triangle_geometry(triangles)
    C,N=_project_bent_by_meta(C0,meta,geom,radius)
    return C,N,A


def generate_smooth_bent_two_arm_dipole_points(
    radius: float,
    total_length: float,
    feed_gap: float,
    depth_frac: float = 0.80,
    bend_radius_over_r: float = 30.0,
    bend_angle_deg: float = 15.0,
    sources_per_arm: int = 900,
    collocation_factor: float = 2.0,
    n_phi: int = 72,
    n_cyl: int = 96,
    n_hemi: int = 24,
    n_pre: int = 34,
    n_arc: int = 58,
    n_post: int = 34,
) -> MASGeometry:
    """Production geometry for the frozen Step-2 MC gentle smooth bend.

    Lower arm is straight; upper arm has one tangent-continuous circular bend.
    This is intentionally restricted to the validated smooth geometry family.
    """
    R=float(radius); total=float(total_length); gap=float(feed_gap)
    frac=float(depth_frac); Larm=0.5*(total-gap)
    Rbend=float(bend_radius_over_r)*R
    angle=np.deg2rad(float(bend_angle_deg))
    if R<=0.0 or gap<=0.0 or Larm<=2.0*R or not (0.0<frac<1.0):
        raise ValueError("Invalid smooth-bent dipole geometry")
    Rs=(1.0-frac)*R
    nsrc=int(sources_per_arm)
    ncoll=max(nsrc+1,int(np.ceil(float(collocation_factor)*nsrc)))
    z_bottom=-0.5*total

    # Frozen MC physical surfaces.
    T0loc=_generate_capsule_triangles(R,R,Larm,n_cyl,n_phi,n_hemi)
    T0=_shift_z(T0loc,z_bottom)
    C00,_,A0=_triangle_geometry(T0loc)
    C0local,N0=_project_to_capsule(C00,R,R,Larm)
    C0=_shift_z(C0local,z_bottom)

    T1,meta1,geom=_generate_bent_capsule_triangles(
        R,R,Larm,gap,Rbend,angle,n_pre,n_arc,n_post,n_phi,n_hemi
    )
    C1,N1,A1=_bent_candidates(T1,meta1,geom,R)

    # Frozen MC physical collocation seeds.
    i0=_fps_indices(C0,ncoll,seed=3101)
    i1=_fps_indices(C1,ncoll,seed=3102)
    OP0,OP1=C0[i0],C1[i1]
    ON0,ON1=N0[i0],N1[i1]
    OW0=_nearest_partition_weights(C0,A0,OP0)
    OW1=_nearest_partition_weights(C1,A1,OP1)

    # Coherent inner surfaces.
    Ts0loc=_generate_capsule_triangles(Rs,R,Larm,n_cyl,n_phi,n_hemi)
    Ts0=_shift_z(Ts0loc,z_bottom)
    SC00,_,SA0=_triangle_geometry(Ts0loc)
    SC0local,SN0=_project_to_capsule(SC00,Rs,R,Larm)
    SC0=_shift_z(SC0local,z_bottom)

    Ts1,smeta1,sgeom=_generate_bent_capsule_triangles(
        Rs,R,Larm,gap,Rbend,angle,n_pre,n_arc,n_post,n_phi,n_hemi
    )
    SC1,SN1,SA1=_bent_candidates(Ts1,smeta1,sgeom,Rs)

    j0=_fps_indices(SC0,nsrc,seed=4101)
    j1=_fps_indices(SC1,nsrc,seed=4102)
    SP0,SP1=SC0[j0],SC1[j1]
    SNS0,SNS1=SN0[j0],SN1[j1]
    SW0=_nearest_partition_weights(SC0,SA0,SP0)
    SW1=_nearest_partition_weights(SC1,SA1,SP1)
    SW0*=float(np.sum(A0))/max(float(np.sum(SW0)),_EPS)
    SW1*=float(np.sum(A1))/max(float(np.sum(SW1)),_EPS)

    # Corresponding physical anchors; not used by the weighted solver but kept
    # consistent with MASGeometry semantics.
    AP0local,_=_project_to_capsule(SC0local[j0],R,R,Larm)
    AP0=_shift_z(AP0local,z_bottom)
    smeta_sel=[smeta1[int(j)] for j in j1]
    AP1,_=_project_bent_by_meta(SP1,smeta_sel,geom,R)

    g=MASGeometry(
        ant_obs_points=np.concatenate((AP0,AP1)),
        ant_src_points=np.concatenate((SP0,SP1)),
        ant_normals=np.concatenate((SNS0,SNS1)),
        source_weights=np.concatenate((SW0,SW1)),
        solve_obs_points=np.concatenate((OP0,OP1)),
        solve_normals=np.concatenate((ON0,ON1)),
        solve_weights=np.concatenate((OW0,OW1)),
        surface_triangles=np.concatenate((T0,T1)),
        metadata={
            "kind":"smooth_bent_two_arm_dipole",
            "validated_scope":"straight_lower_plus_single_gentle_C1_bent_upper",
            "radius":R,"total_length":total,"feed_gap":gap,"arm_length":Larm,
            "depth_frac":frac,"source_offset":frac*R,"inner_radius":Rs,
            "bend_radius_over_r":float(bend_radius_over_r),
            "bend_angle_deg":float(bend_angle_deg),
            "sources_per_arm":nsrc,"collocation_points_per_arm":ncoll,
            "collocation_factor":float(collocation_factor),
            "n_phi":int(n_phi),"n_cyl":int(n_cyl),"n_hemi":int(n_hemi),
            "n_pre":int(n_pre),"n_arc":int(n_arc),"n_post":int(n_post),
            "source_sampling":"global_fps_independently_per_arm",
            "collocation_sampling":"independent_global_fps_per_arm",
            "source_weighting":"inner_voronoi_rescaled_to_physical_arm_area",
            "centerline_continuity":"C1_tangent_continuous_curvature_jumps_at_joins",
        },
    )
    g.source_tangent1,g.source_tangent2=build_local_tangent_frames(
        np.concatenate((SNS0,SNS1))
    )
    return g

def generate_spherical_antenna_points(
    radius: float,
    center=(0.0, 0.0, 0.0),
    n_theta: int = 12,
    n_phi: int = 24,
    source_offset: Optional[float] = None,
) -> MASGeometry:
    """Generate a spherical PEC point cloud for regression tests or probes."""
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    n_theta = int(n_theta)
    n_phi = int(n_phi)
    if n_theta < 3 or n_phi < 6:
        raise ValueError("Use n_theta >= 3 and n_phi >= 6")
    if source_offset is None:
        source_offset = 0.35 * radius
    source_offset = float(source_offset)
    if not (0.0 < source_offset < radius):
        raise ValueError("source_offset must be inside the conductor: 0 < source_offset < radius")

    center = _as_vec3(center, "center")
    obs = []
    src = []
    normals = []
    weights = []
    dtheta = np.pi / n_theta
    dphi = 2.0 * np.pi / n_phi
    for it in range(n_theta):
        theta = (it + 0.5) * dtheta
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        band_area = radius * radius * sin_t * dtheta * dphi
        for ip in range(n_phi):
            phi = (ip + 0.5) * dphi
            normal = np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], dtype=np.float64)
            obs.append(center + radius * normal)
            src.append(center + (radius - source_offset) * normal)
            normals.append(normal)
            weights.append(band_area)
    return MASGeometry(
        ant_obs_points=np.asarray(obs, dtype=np.float64),
        ant_src_points=np.asarray(src, dtype=np.float64),
        ant_normals=np.asarray(normals, dtype=np.float64),
        source_weights=np.asarray(weights, dtype=np.float64),
        metadata={
            "kind": "sphere",
            "radius": radius,
            "center": center.tolist(),
            "n_theta": n_theta,
            "n_phi": n_phi,
            "source_offset": source_offset,
        },
    )


def _load_stl_triangles_with_trimesh(path: str | Path):
    try:
        import trimesh  # optional dependency
    except Exception:
        return None
    mesh = trimesh.load_mesh(str(path), force="mesh")
    if hasattr(mesh, "geometry"):
        # Scene: concatenate all mesh geometries.
        meshes = [g for g in mesh.geometry.values() if hasattr(g, "triangles")]
        if not meshes:
            raise ValueError(f"No triangle mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(meshes)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError(f"STL file {path} did not contain usable triangles")
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    return triangles, normals, areas, {"loader": "trimesh", "watertight": bool(getattr(mesh, "is_watertight", False))}


def _load_ascii_stl_triangles(path: str | Path):
    """Tiny fallback parser for ASCII STL when trimesh is unavailable."""
    vertices = []
    normals = []
    current_normal = None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) >= 5 and parts[0].lower() == "facet" and parts[1].lower() == "normal":
                current_normal = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
            elif len(parts) >= 4 and parts[0].lower() == "vertex":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(vertices) % 3 == 0:
                    normals.append(current_normal if current_normal is not None else np.zeros(3, dtype=np.float64))
    if len(vertices) < 3 or len(vertices) % 3 != 0:
        raise ImportError(
            "trimesh is not installed and the STL file is not a readable ASCII STL. "
            "Install trimesh for binary STL and robust industrial mesh import."
        )
    triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    v1 = triangles[:, 1] - triangles[:, 0]
    v2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(v1, v2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    bad = np.linalg.norm(normals, axis=1) <= _EPS
    normals[bad] = cross[bad]
    return triangles, normals, areas, {"loader": "ascii_stl_fallback", "watertight": None}


def _subdivide_triangles_to_max_edge(triangles: np.ndarray, max_edge: float, max_levels: int = 6) -> np.ndarray:
    """Uniformly split only triangles whose longest edge exceeds ``max_edge``.

    This does not change the represented piecewise-planar surface; it only
    provides a geometry-aware source/collocation discretization for imported STL
    meshes whose original facets are too large or highly anisotropic.
    """
    tri = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    max_edge = float(max_edge)
    if max_edge <= 0.0:
        return tri
    for _ in range(max(0, int(max_levels))):
        e01 = np.linalg.norm(tri[:,1]-tri[:,0], axis=1)
        e12 = np.linalg.norm(tri[:,2]-tri[:,1], axis=1)
        e20 = np.linalg.norm(tri[:,0]-tri[:,2], axis=1)
        split = np.maximum.reduce([e01,e12,e20]) > max_edge
        if not np.any(split):
            break
        out = []
        for t, do_split in zip(tri, split):
            if not do_split:
                out.append(t)
                continue
            a,b,c = t
            ab=0.5*(a+b); bc=0.5*(b+c); ca=0.5*(c+a)
            out.extend([
                np.asarray([a,ab,ca]),
                np.asarray([ab,b,bc]),
                np.asarray([ca,bc,c]),
                np.asarray([ab,bc,ca]),
            ])
        tri = np.asarray(out, dtype=np.float64)
    return tri


def generate_stl_antenna_points(
    stl_path: str | Path,
    source_offset: Optional[float] = None,
    max_points: Optional[int] = None,
    scale: float = 1.0,
    translation=(0.0, 0.0, 0.0),
    transform: Optional[np.ndarray] = None,
    flip_normals: bool = False,
    min_area: float = 0.0,
    patch_min_points: int = 0,
    patch_normal_axis=(0.0, 0.0, 1.0),
    patch_normal_threshold: float = 0.75,
    target_patch_size: Optional[float] = None,
    max_subdivision_levels: int = 6,
) -> MASGeometry:
    """Generate MAS points from an STL triangle mesh.

    Observation points are triangle centroids or deterministic area-stratified
    face samples.  Auxiliary sources are shifted inward along ``-normal`` by
    ``source_offset``.  ``trimesh`` is used when available; a simple ASCII STL
    fallback is provided for small tests, but industrial binary STL should use
    trimesh.
    """
    stl_path = Path(stl_path)
    loaded = None
    # Avoid importing trimesh for tiny ASCII STL test fixtures; trimesh remains
    # the robust path for binary and industrial meshes.
    try:
        prefix = stl_path.read_bytes()[:512].lstrip().lower()
        if prefix.startswith(b"solid") and b"facet" in prefix:
            loaded = _load_ascii_stl_triangles(stl_path)
    except Exception:
        loaded = None
    if loaded is None:
        loaded = _load_stl_triangles_with_trimesh(stl_path)
    if loaded is None:
        loaded = _load_ascii_stl_triangles(stl_path)
    triangles, normals, areas, load_meta = loaded

    scale = float(scale)
    translation = _as_vec3(translation, "translation")
    triangles = triangles * scale
    if transform is not None:
        T = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        pts_h = np.concatenate([triangles.reshape(-1, 3), np.ones((triangles.size // 3, 1))], axis=1)
        triangles = (pts_h @ T.T)[:, :3].reshape(-1, 3, 3)
        # Recompute normals after arbitrary transform.
        v1 = triangles[:, 1] - triangles[:, 0]
        v2 = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(v1, v2)
        areas = 0.5 * np.linalg.norm(normals, axis=1)
    triangles = triangles + translation[None, None, :]

    if target_patch_size is not None and float(target_patch_size) > 0.0:
        triangles = _subdivide_triangles_to_max_edge(
            triangles, float(target_patch_size), max_levels=int(max_subdivision_levels)
        )
        # Recompute all geometric quantities after subdivision.
        normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        areas = 0.5*np.linalg.norm(normals, axis=1)

    v1 = triangles[:, 1] - triangles[:, 0]
    v2 = triangles[:, 2] - triangles[:, 0]
    geometric_normals = np.cross(v1, v2)
    geometric_areas = 0.5 * np.linalg.norm(geometric_normals, axis=1)
    if areas is None or len(areas) != len(geometric_areas):
        areas = geometric_areas
    valid = np.asarray(areas, dtype=np.float64) > max(0.0, float(min_area))
    valid &= geometric_areas > _EPS
    if not np.any(valid):
        raise ValueError("STL mesh has no positive-area triangles after filtering")
    triangles = triangles[valid]
    areas = np.asarray(areas, dtype=np.float64)[valid]
    normals = np.asarray(normals, dtype=np.float64)[valid]
    geometric_normals = geometric_normals[valid]

    normal_norm = np.linalg.norm(normals, axis=1)
    bad_normals = normal_norm <= _EPS
    normals[bad_normals] = geometric_normals[bad_normals]
    normal_norm = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(normal_norm[:, None], _EPS)
    if flip_normals:
        normals = -normals

    total_area = float(np.sum(areas))
    centers = np.mean(triangles, axis=1)
    weights = areas.copy()

    original_num_triangles = int(len(centers))
    full_triangles = triangles.copy()
    full_centers = centers.copy()
    full_normals = normals.copy()
    full_areas = areas.copy()
    patch_min_points = max(0, int(patch_min_points))
    patch_axis = _normalize(patch_normal_axis, "patch_normal_axis")
    patch_normal_threshold = min(1.0, max(0.0, float(patch_normal_threshold)))

    def _area_stratified_select(pool_idx: np.ndarray, n_select: int) -> tuple[np.ndarray, np.ndarray]:
        pool_idx = np.asarray(pool_idx, dtype=np.int64)
        n_select = int(n_select)
        if n_select <= 0 or len(pool_idx) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        if n_select >= len(pool_idx):
            return pool_idx.copy(), areas[pool_idx].copy()
        local_areas = areas[pool_idx]
        cumulative = np.cumsum(local_areas)
        subtotal = float(cumulative[-1])
        targets = (np.arange(n_select, dtype=np.float64) + 0.5) * subtotal / n_select
        local = np.searchsorted(cumulative, targets, side="left")
        selected = pool_idx[np.clip(local, 0, len(pool_idx) - 1)]
        return selected, np.full(n_select, subtotal / n_select, dtype=np.float64)

    feature_measure = np.abs(normals @ patch_axis)
    feature_mask = feature_measure >= patch_normal_threshold
    feature_count_before_sampling = int(np.count_nonzero(feature_mask))

    patch_ids = np.full(len(centers), -1, dtype=np.int64)
    patch_info = []
    if patch_min_points > 0 and feature_count_before_sampling > 0:
        feature_idx_all = np.where(feature_mask)[0]
        # Patch definition for end/feed faces of a z-axis dipole:
        # group by normal sign and approximate z-plane.  This separates
        # outer end caps and the two feed-gap faces.
        signs = np.sign(normals[feature_idx_all] @ patch_axis)
        zvals = centers[feature_idx_all, 2]
        rounded_z = np.round(zvals, decimals=12)
        keys = sorted(set((int(s), float(z)) for s, z in zip(signs, rounded_z)))
        for pid, key in enumerate(keys):
            s, z = key
            members = feature_idx_all[(signs == s) & (rounded_z == z)]
            patch_ids[members] = pid
            patch_info.append({
                "patch_id": int(pid),
                "normal_sign": int(s),
                "z": float(z),
                "num_faces_before_sampling": int(len(members)),
            })

    if max_points is not None and int(max_points) > 0 and int(max_points) < len(centers):
        max_points = int(max_points)

        if patch_min_points > 0 and len(patch_info) > 0:
            selected_parts = []
            weight_parts = []
            used = np.zeros(len(centers), dtype=bool)

            # First, give every identified patch its minimum quota, limited by
            # both available faces and the total budget.  This is scheme 3:
            # each important patch gets a guaranteed minimum representation.
            remaining_budget = max_points
            for info in patch_info:
                pid = info["patch_id"]
                members = np.where(patch_ids == pid)[0]
                quota = min(patch_min_points, len(members), remaining_budget)
                sel, w = _area_stratified_select(members, quota)
                selected_parts.append(sel)
                weight_parts.append(w)
                used[sel] = True
                remaining_budget -= len(sel)
                info["num_faces_after_sampling"] = int(len(sel))
                if remaining_budget <= 0:
                    break

            # Then spend the remaining budget on all non-selected side-wall
            # faces by deterministic area-stratified sampling.  We place the
            # side-wall samples first in the final ordering, then patch samples,
            # so the sorted heatmap can mark the row split cleanly.
            regular_pool = np.where(~used & ~feature_mask)[0]
            regular_sel, regular_w = _area_stratified_select(regular_pool, remaining_budget)

            if len(regular_sel) > 0:
                selected = np.concatenate([regular_sel] + selected_parts)
                weights = np.concatenate([regular_w] + weight_parts)
            else:
                selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
                weights = np.concatenate(weight_parts) if weight_parts else np.empty(0, dtype=np.float64)

            centers = centers[selected]
            normals = normals[selected]
        else:
            cumulative = np.cumsum(areas)
            targets = (np.arange(max_points, dtype=np.float64) + 0.5) * total_area / max_points
            idx = np.searchsorted(cumulative, targets, side="left")
            idx = np.clip(idx, 0, len(centers) - 1)
            centers = centers[idx]
            normals = normals[idx]
            weights = np.full(max_points, total_area / max_points, dtype=np.float64)

    feature_count_after_sampling = int(np.count_nonzero(np.abs(normals @ patch_axis) >= patch_normal_threshold))

    if source_offset is None:
        bbox_min = np.min(triangles.reshape(-1, 3), axis=0)
        bbox_max = np.max(triangles.reshape(-1, 3), axis=0)
        source_offset = 0.02 * float(np.linalg.norm(bbox_max - bbox_min))
    source_offset = float(source_offset)
    if source_offset <= 0.0:
        raise ValueError("source_offset must be positive for STL MAS geometry")
    if load_meta.get("watertight") is False:
        warnings.warn("STL mesh is not watertight; verify normal orientation before using inward MAS offsets.", RuntimeWarning, stacklevel=2)

    obs = centers
    src = centers - source_offset * normals
    return MASGeometry(
        ant_obs_points=obs,
        ant_src_points=src,
        ant_normals=normals,
        source_weights=weights,
        surface_triangles=full_triangles,
        surface_triangle_normals=full_normals,
        surface_triangle_areas=full_areas,
        metadata={
            "kind": "stl",
            "stl_path": str(stl_path),
            "num_triangles_original": original_num_triangles,
            "num_triangles_used": int(len(obs)),
            "source_offset": source_offset,
            "scale": scale,
            "translation": translation.tolist(),
            "flip_normals": bool(flip_normals),
            "surface_area_estimate": total_area,
            "patch_min_points": int(patch_min_points),
            "patch_normal_axis": patch_axis.tolist(),
            "patch_normal_threshold": patch_normal_threshold,
            "target_patch_size": None if target_patch_size is None else float(target_patch_size),
            "max_subdivision_levels": int(max_subdivision_levels),
            "feature_count_before_sampling": feature_count_before_sampling,
            "feature_count_after_sampling": feature_count_after_sampling,
            "patch_info": patch_info,
            **load_meta,
        },
    )


def configure_oversampled_collocation(geometry: MASGeometry, oversample: int = 2) -> MASGeometry:
    """Attach a denser physical-surface collocation cloud for the Step-2G solve.

    The source patches and their weights are unchanged.  Only the PEC testing
    points are oversampled.  Cylinder/sphere use phase-shifted parameter-space
    sets; STL uses deterministic area-stratified barycentric samples of the
    original triangle surface.
    """
    oversample = max(1, int(oversample))
    if oversample == 1:
        return geometry.set_solve_collocation(geometry.ant_obs_points, geometry.ant_normals, geometry.source_weights)

    meta = dict(geometry.metadata)
    kind = str(meta.get("kind", "")).lower()
    pts = []
    nrms = []
    if kind == "cylinder":
        radius = float(meta["radius"]); length = float(meta["length"])
        center = _as_vec3(meta.get("center", (0,0,0)), "center")
        ex, ey, ez = _orthonormal_basis(meta.get("axis", (0,0,1)))
        nz = int(meta["n_length"]); nphi = int(meta["n_phi"])
        for q in range(oversample):
            zs = (0.371*q) % 0.49
            ps = (0.293*q) % 1.0
            zvals = ((np.arange(nz)+0.5+zs)/nz-0.5)*length
            phis = 2*np.pi*(np.arange(nphi)+0.5+ps)/nphi
            for z in zvals:
                axial = center + z*ez
                for phi in phis:
                    nvec = np.cos(phi)*ex + np.sin(phi)*ey
                    pts.append(axial + radius*nvec); nrms.append(nvec)
    elif kind == "sphere":
        radius = float(meta["radius"]); center = _as_vec3(meta.get("center", (0,0,0)), "center")
        nt = int(meta["n_theta"]); np_ = int(meta["n_phi"])
        for q in range(oversample):
            ts = (0.317*q) % 0.45
            ps = (0.271*q) % 1.0
            thetas = np.pi*(np.arange(nt)+0.5+ts)/nt
            phis = 2*np.pi*(np.arange(np_)+0.5+ps)/np_
            for th in thetas:
                st, ct = np.sin(th), np.cos(th)
                for ph in phis:
                    nvec = np.array([st*np.cos(ph), st*np.sin(ph), ct])
                    pts.append(center + radius*nvec); nrms.append(nvec)
    elif kind == "stl" and geometry.surface_triangles is not None:
        tris = np.asarray(geometry.surface_triangles, dtype=np.float64)
        fn = np.asarray(geometry.surface_triangle_normals, dtype=np.float64)
        fa = np.asarray(geometry.surface_triangle_areas, dtype=np.float64)
        total = float(np.sum(fa))
        target_n = oversample * geometry.num_sources
        cumulative = np.cumsum(fa)
        targets = (np.arange(target_n, dtype=np.float64)+0.5)*total/target_n
        ids = np.searchsorted(cumulative, targets, side="left")
        ids = np.clip(ids, 0, len(tris)-1)
        bary = np.asarray([[0.60,0.20,0.20],[0.20,0.60,0.20],[0.20,0.20,0.60]], dtype=np.float64)
        for q, tid in enumerate(ids):
            b = bary[q % len(bary)]
            pts.append(b @ tris[tid]); nrms.append(fn[tid])
    else:
        raise ValueError(f"Oversampled collocation is not implemented for geometry kind={kind!r}")

    points = np.asarray(pts, dtype=np.float64)
    normals = np.asarray(nrms, dtype=np.float64)
    total_area = float(np.sum(geometry.source_weights))
    weights = np.full(len(points), total_area/len(points), dtype=np.float64)
    geometry.set_solve_collocation(points, normals, weights)
    geometry.metadata = dict(geometry.metadata)
    geometry.metadata.update({
        "collocation_oversample": oversample,
        "num_source_points": geometry.num_sources,
        "num_solve_collocation_points": geometry.num_collocation_points,
    })
    return geometry


def _assemble_dipole_blocks(geometry: MASGeometry, obs_points: np.ndarray, dt: float, epsilon0: float, c: float, softening_distance: float):
    obs = np.asarray(obs_points, dtype=np.float64).reshape(-1, 3)
    src = np.asarray(geometry.ant_src_points, dtype=np.float64).reshape(-1, 3)
    nsrc = len(src); nobs = len(obs)
    ce = 1.0/(4.0*np.pi*epsilon0)
    I = np.eye(3, dtype=np.float64)
    Gp = np.zeros((3*nobs, 3*nsrc), dtype=np.float64)
    Gk = np.zeros_like(Gp); Gd = np.zeros_like(Gp)
    for j in range(nsrc):
        r = obs-src[j][None, :]
        R = np.sqrt(np.sum(r*r, axis=1)+softening_distance*softening_distance)
        R = np.maximum(R, _EPS)
        rr = r[:, :, None]*r[:, None, :]
        w = float(geometry.source_weights[j])
        bp = ce*w*(3.0*rr/R[:,None,None]**5-I[None,:,:]/R[:,None,None]**3)
        bk = ce*w*(3.0*rr/R[:,None,None]**4-I[None,:,:]/R[:,None,None]**2)/c
        bd = ce*w*(rr/R[:,None,None]**3-I[None,:,:]/R[:,None,None])/(c*c)
        Gp[:,3*j:3*j+3] = bp.reshape(3*nobs,3)
        Gk[:,3*j:3*j+3] = bk.reshape(3*nobs,3)
        Gd[:,3*j:3*j+3] = bd.reshape(3*nobs,3)
    return Gp, Gk, Gd



def _estimate_smax_power(A: np.ndarray, niter: int = 8) -> float:
    """Deterministic Step-2M spectral-scale estimate used for ridge strength."""
    A = np.asarray(A, dtype=np.float64)
    rng = np.random.default_rng(1234)
    x = rng.normal(size=A.shape[1])
    x /= max(float(np.linalg.norm(x)), _EPS)
    for _ in range(int(niter)):
        y = A @ x
        z = A.T @ y
        nz = float(np.linalg.norm(z))
        if nz < _EPS:
            return 1.0
        x = z / nz
    return float(np.linalg.norm(A @ x))


def assemble_impedance_matrix(
    geometry: MASGeometry,
    dt: float,
    epsilon0: float = 8.854187817e-12,
    c: float = 299792458.0,
    softening_distance: Optional[float] = None,
    normal_regularization: float = 1.0e-3,
    diagonal_loading: float = 1.0e-10,
    use_pinv: bool = True,
    spatial_solver: str = "legacy_cartesian_square",
    tikhonov_rel: float = 1.0e-4,
    svd_rcond: float = 1.0e-10,
) -> MASGeometry:
    """Assemble the Step-2B continuity-consistent local electric-dipole operator.

    Each MAS coefficient is a surface-current density ``K`` [A/m] attached to a
    source patch of area ``w``.  The lumped electric current moment is therefore
    ``m = w K`` [A m], while the corresponding dipole moment is ``p = w P``
    with ``dP/dt = K``.

    At time node n the temporal basis is

        P^n = P^(n-1) + dt/2 (K^(n-1) + K^n),
        Kdot^n = (K^n - K^(n-1))/dt.

    The exact (time-local, zero-retardation) electric-dipole field can therefore
    be split into a matrix multiplying the current unknown K^n and a known
    history field.  The matrix assembled here contains *all three* dipole
    electric-field terms: 1/R^3 electrostatic, 1/R^2 induction, and 1/R
    radiation.  This replaces the legacy vector-potential-only J/dt surrogate.
    """
    dt = float(dt)
    epsilon0 = float(epsilon0)
    c = float(c)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if softening_distance is None:
        offsets = np.linalg.norm(geometry.ant_obs_points - geometry.ant_src_points, axis=1)
        softening_distance = 0.25 * float(np.median(offsets)) if len(offsets) else 0.0
    softening_distance = max(float(softening_distance), 0.0)
    normal_regularization = max(float(normal_regularization), 0.0)
    diagonal_loading = max(float(diagonal_loading), 0.0)

    spatial_solver = str(spatial_solver).lower()
    tikhonov_rel = max(float(tikhonov_rel), 0.0)
    svd_rcond = max(float(svd_rcond), 0.0)
    n = geometry.num_sources

    if spatial_solver in ("tangential_oversampled", "tangent_ls", "step2g", "tangential_weighted_scaled", "step3a"):
        obs = geometry.solve_obs_points
        G_p, G_k, G_kdot = _assemble_dipole_blocks(
            geometry, obs, dt, epsilon0, c, softening_distance
        )
        G_physical = 0.5*dt*G_p + G_k + G_kdot/dt
        B = _basis_matrix(geometry.source_tangent1, geometry.source_tangent2)
        C = _constraint_matrix(geometry.solve_tangent1, geometry.solve_tangent2)
        A = C @ G_physical @ B

        weighted_scaled = spatial_solver in ("tangential_weighted_scaled", "step3a")
        if weighted_scaled:
            # Step-2M validated objective:
            #   1) area-weight tangential equations,
            #   2) normalize every unknown column,
            #   3) ridge with lambda_rel * smax(A_scaled).
            point_w = np.asarray(geometry.solve_weights, dtype=np.float64)
            row_scale = np.repeat(
                np.sqrt(point_w / max(float(np.mean(point_w)), _EPS)), 2
            )
            Aw = A * row_scale[:, None]
            col_scale = np.linalg.norm(Aw, axis=0)
            col_scale = np.where(col_scale > _EPS, col_scale, 1.0)
            As = Aw / col_scale[None, :]

            # Reproduce the frozen Step-2 MB1/MC/MD2 ridge path exactly:
            # power-estimated smax, normal equations, and Cholesky factorization.
            # An SVD ridge is mathematically similar but is not implementation-
            # equivalent enough for the Step 3-A production/reference regression.
            from scipy.linalg import cho_factor, cho_solve
            smax = max(_estimate_smax_power(As, niter=8), _EPS)
            alpha = tikhonov_rel*smax
            H = As.T @ As
            H.flat[::H.shape[0] + 1] += alpha*alpha
            cf = cho_factor(H, lower=True, overwrite_a=True, check_finite=False)

            # Explicit production response map:
            # x = diag(1/col_scale) H^-1 As^T diag(row_scale) b.
            solver_scaled = cho_solve(cf, As.T, check_finite=False)
            solver = (solver_scaled * row_scale[None, :]) / col_scale[:, None]

            # Avoid a second full SVD solely for diagnostics.
            condition_number = None
            solver_name = "tangential_weighted_scaled"
        else:
            U, sval, Vt = np.linalg.svd(A, full_matrices=False)
            smax = max(float(sval[0]) if len(sval) else 0.0, _EPS)
            alpha = tikhonov_rel*smax
            if alpha > 0.0:
                filt = sval/(sval*sval+alpha*alpha)
            else:
                filt = np.where(sval > svd_rcond*smax, 1.0/sval, 0.0)
            solver = Vt.T @ (filt[:,None]*U.T)
            condition_number = smax/max(float(sval[-1]) if len(sval) else 0.0, _EPS)
            row_scale = np.ones(A.shape[0], dtype=np.float64)
            col_scale = np.ones(A.shape[1], dtype=np.float64)
            solver_name = "tangential_oversampled"

        geometry.G_p = G_p
        geometry.G_k = G_k
        geometry.G_kdot = G_kdot
        geometry.G_physical = G_physical
        geometry.G = A
        geometry.G_inv = solver
        geometry.metadata = dict(geometry.metadata)
        geometry.metadata.update({
            "spatial_solver": solver_name,
            "source_dof_per_patch": 2,
            "collocation_equations_per_point": 2,
            "num_source_points": geometry.num_sources,
            "num_collocation_points": geometry.num_collocation_points,
            "num_unknowns": int(2*geometry.num_sources),
            "num_tangential_equations": int(2*geometry.num_collocation_points),
            "tikhonov_rel": tikhonov_rel,
            "svd_rcond": svd_rcond,
            "condition_number_G": condition_number,
            "dt_for_G": dt,
            "epsilon0_for_G": epsilon0,
            "c_for_G": c,
            "softening_distance_for_G": softening_distance,
            "mas_source_model": "continuity_consistent_electric_dipole",
            "mas_unknown": "two_local_tangential_surface_current_density_coefficients_A_per_m",
            "local_temporal_basis": "P^n=P^(n-1)+dt/2(K^(n-1)+K^n); Kdot^n=(K^n-K^(n-1))/dt",
            "local_G_terms": "1/R^3 dipole + 1/R^2 induction + 1/R radiation",
            "row_weighting": "sqrt(solve_weight/mean_solve_weight)" if weighted_scaled else "none",
            "column_scaling": "weighted_operator_l2_norm" if weighted_scaled else "none",
            "ridge_definition": "lambda_rel*smax(weighted_column_scaled_operator)" if weighted_scaled else "lambda_rel*smax(raw_operator)",
        })
        return geometry

    if spatial_solver not in ("legacy_cartesian_square", "legacy", "cartesian_square"):
        raise ValueError(f"Unsupported MAS spatial_solver: {spatial_solver}")

    coeff_e = 1.0 / (4.0 * np.pi * epsilon0)
    identity = np.eye(3, dtype=np.float64)
    G_p = np.zeros((3*n, 3*n), dtype=np.float64)
    G_k = np.zeros_like(G_p)
    G_kdot = np.zeros_like(G_p)

    for i in range(n):
        for j in range(n):
            r = geometry.ant_obs_points[i] - geometry.ant_src_points[j]
            r2 = float(np.dot(r, r) + softening_distance * softening_distance)
            R = max(float(np.sqrt(r2)), _EPS)
            rr = np.outer(r, r)
            w = float(geometry.source_weights[j])
            Bp = coeff_e * w * (3.0 * rr / R**5 - identity / R**3)
            Bk = coeff_e * w * (3.0 * rr / R**4 - identity / R**2) / c
            Bkd = coeff_e * w * (rr / R**3 - identity / R) / (c*c)
            sl_i = slice(3*i, 3*i+3)
            sl_j = slice(3*j, 3*j+3)
            G_p[sl_i, sl_j] = Bp
            G_k[sl_i, sl_j] = Bk
            G_kdot[sl_i, sl_j] = Bkd

    G_physical = 0.5 * dt * G_p + G_k + (1.0 / dt) * G_kdot
    G = np.zeros_like(G_physical)
    for i in range(n):
        normal = geometry.ant_normals[i]
        Pt = identity - np.outer(normal, normal)
        Preg = Pt + normal_regularization * np.outer(normal, normal)
        G[3*i:3*i+3, :] = Preg @ G_physical[3*i:3*i+3, :]

    if diagonal_loading > 0.0:
        scale = max(float(np.linalg.norm(G, ord=np.inf)), 1.0)
        G = G + (diagonal_loading * scale) * np.eye(3*n, dtype=np.float64)

    G_inv = np.linalg.pinv(G, rcond=svd_rcond) if use_pinv else np.linalg.inv(G)
    geometry.G_p = G_p
    geometry.G_k = G_k
    geometry.G_kdot = G_kdot
    geometry.G_physical = G_physical
    geometry.G = G
    geometry.G_inv = G_inv
    try:
        condition_number = float(np.linalg.cond(G))
    except Exception:
        condition_number = float("inf")
    geometry.metadata = dict(geometry.metadata)
    geometry.metadata.update({
        "spatial_solver": "legacy_cartesian_square",
        "dt_for_G": dt,
        "epsilon0_for_G": epsilon0,
        "c_for_G": c,
        "softening_distance_for_G": softening_distance,
        "normal_regularization": normal_regularization,
        "diagonal_loading": diagonal_loading,
        "condition_number_G": condition_number,
        "mas_source_model": "continuity_consistent_electric_dipole",
        "mas_unknown": "surface_current_density_K_A_per_m",
        "lumped_current_moment": "m_j = source_weight_j * K_j [A m]",
        "local_temporal_basis": "P^n=P^(n-1)+dt/2(K^(n-1)+K^n); Kdot^n=(K^n-K^(n-1))/dt",
        "local_G_terms": "1/R^3 dipole + 1/R^2 induction + 1/R radiation",
    })
    return geometry


def build_mas_geometry(kind: str = "cylinder", **kwargs) -> MASGeometry:
    """Factory used by examples and compatibility wrappers."""
    kind = str(kind).lower()
    if kind in ("parallel_two_rounded_dipoles", "dual_parallel_dipoles", "md2_dual"):
        return generate_parallel_two_rounded_dipoles_points(**kwargs)
    if kind in ("smooth_bent_two_arm_dipole", "smooth_bent_dipole", "gentle_bent_dipole"):
        return generate_smooth_bent_two_arm_dipole_points(**kwargs)
    if kind in ("rounded_two_arm_dipole", "rounded_dipole", "smooth_dipole"):
        return generate_rounded_two_arm_dipole_points(**kwargs)
    if kind in ("wire", "dipole", "cylinder", "cylindrical"):
        return generate_cylindrical_antenna_points(**kwargs)
    if kind in ("sphere", "spherical"):
        return generate_spherical_antenna_points(**kwargs)
    if kind in ("stl", "mesh", "tri_mesh", "triangle_mesh"):
        return generate_stl_antenna_points(**kwargs)
    raise ValueError(f"Unsupported MAS geometry kind: {kind}")


def preprocess_mas_antenna(kind: str = "cylinder", save_path: Optional[str | Path] = None, **kwargs) -> MASGeometry:
    """High-level preprocessing entry point.

    Expected keyword groups:
      - geometry parameters: radius, length, center, axis, n_length, n_phi, ...
      - matrix parameters: dt, epsilon0, c, softening_distance, ...
    """
    matrix_keys = {
        "dt", "epsilon0", "c", "softening_distance", "normal_regularization",
        "diagonal_loading", "use_pinv", "spatial_solver", "tikhonov_rel", "svd_rcond",
    }
    matrix_kwargs = {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in matrix_keys}
    if "dt" not in matrix_kwargs:
        raise ValueError("preprocess_mas_antenna requires dt so G can be assembled")
    collocation_oversample = int(kwargs.pop("collocation_oversample", 1))
    geometry = build_mas_geometry(kind=kind, **kwargs)
    solver_name = str(matrix_kwargs.get("spatial_solver", "legacy_cartesian_square")).lower()
    if solver_name in ("tangential_oversampled", "tangent_ls", "step2g", "tangential_weighted_scaled", "step3a"):
        if str(geometry.metadata.get("kind", "")).lower() not in (
            "rounded_two_arm_dipole", "smooth_bent_two_arm_dipole",
            "parallel_two_rounded_dipoles"
        ):
            geometry = configure_oversampled_collocation(geometry, collocation_oversample)
    geometry = assemble_impedance_matrix(geometry, **matrix_kwargs)
    if save_path is not None:
        geometry.save_npz(save_path)
    return geometry
