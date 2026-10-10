
"""Generate a simple center-fed cylindrical dipole antenna as an ASCII STL.

The model is deliberately simple and reproducible:
- two coaxial PEC cylindrical arms,
- a small feed gap at the origin,
- radially subdivided closed end caps,
- outward-oriented triangle normals.

The radial cap subdivision is important for MAS figures and preprocessing:
a single fan cap has all triangle centroids on one ring, which can make the
feed face look undersampled even when many triangles exist.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import numpy as np


def _normal_of_triangle(tri: np.ndarray) -> np.ndarray:
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = np.linalg.norm(n)
    if norm <= 1.0e-300:
        return np.zeros(3)
    return n / norm


def _disk_point(radius: float, z: float, r_frac: float, phi: float) -> np.ndarray:
    r = radius * r_frac
    return np.array([r * math.cos(phi), r * math.sin(phi), z], dtype=float)


def _add_cap(triangles, z: float, radius: float, n_phi: int, n_r_cap: int, normal_sign: int) -> None:
    """Add a radially subdivided disk cap.

    normal_sign = +1 gives +z normals; normal_sign = -1 gives -z normals.
    """
    n_r_cap = max(1, int(n_r_cap))
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi + 1)

    for ir in range(n_r_cap):
        r0 = ir / n_r_cap
        r1 = (ir + 1) / n_r_cap
        for ip in range(n_phi):
            p_i0 = _disk_point(radius, z, r0, phi[ip])
            p_i1 = _disk_point(radius, z, r0, phi[ip + 1])
            p_o0 = _disk_point(radius, z, r1, phi[ip])
            p_o1 = _disk_point(radius, z, r1, phi[ip + 1])

            if ir == 0:
                # One triangle from center to the first radial ring.
                if normal_sign > 0:
                    tri = np.array([p_i0, p_o0, p_o1])
                else:
                    tri = np.array([p_i0, p_o1, p_o0])
                triangles.append(tri)
            else:
                # Annular quad split into two triangles.
                if normal_sign > 0:
                    triangles.append(np.array([p_i0, p_o0, p_o1]))
                    triangles.append(np.array([p_i0, p_o1, p_i1]))
                else:
                    triangles.append(np.array([p_i0, p_o1, p_o0]))
                    triangles.append(np.array([p_i0, p_i1, p_o1]))


def _add_cylinder_segment(
    triangles,
    z0: float,
    z1: float,
    radius: float,
    n_phi: int,
    n_z: int,
    n_r_cap: int,
) -> None:
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi + 1)
    z = np.linspace(z0, z1, n_z + 1)

    def p(kz, kp):
        return np.array([
            radius * math.cos(phi[kp]),
            radius * math.sin(phi[kp]),
            z[kz],
        ], dtype=float)

    # Side wall. Vertex order chosen for outward normals.
    for iz in range(n_z):
        for ip in range(n_phi):
            p00 = p(iz, ip)
            p01 = p(iz, ip + 1)
            p10 = p(iz + 1, ip)
            p11 = p(iz + 1, ip + 1)
            triangles.append(np.array([p00, p01, p11]))
            triangles.append(np.array([p00, p11, p10]))

    # Bottom cap outward normal -z; top cap outward normal +z.
    _add_cap(triangles, z0, radius, n_phi, n_r_cap, normal_sign=-1)
    _add_cap(triangles, z1, radius, n_phi, n_r_cap, normal_sign=+1)


def generate_center_fed_dipole(
    output: str | Path,
    *,
    total_length: float = 0.060,
    radius: float = 0.0010,
    feed_gap: float = 0.0020,
    n_phi: int = 40,
    n_z_per_arm: int = 10,
    n_r_cap: int = 5,
) -> Path:
    """Write a two-arm cylindrical dipole STL and return its path."""
    output = Path(output)
    if total_length <= feed_gap:
        raise ValueError("total_length must be larger than feed_gap")
    if radius <= 0.0 or feed_gap <= 0.0:
        raise ValueError("radius and feed_gap must be positive")
    triangles = []
    half = 0.5 * total_length
    gap_half = 0.5 * feed_gap
    _add_cylinder_segment(triangles, -half, -gap_half, radius, n_phi, n_z_per_arm, n_r_cap)
    _add_cylinder_segment(triangles, gap_half, half, radius, n_phi, n_z_per_arm, n_r_cap)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("solid center_fed_dipole\n")
        for tri in triangles:
            n = _normal_of_triangle(tri)
            f.write(f"  facet normal {n[0]:.12e} {n[1]:.12e} {n[2]:.12e}\n")
            f.write("    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]:.12e} {v[1]:.12e} {v[2]:.12e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid center_fed_dipole\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="center_fed_dipole.stl")
    parser.add_argument("--total-length", type=float, default=0.060)
    parser.add_argument("--radius", type=float, default=0.0010)
    parser.add_argument("--feed-gap", type=float, default=0.0020)
    parser.add_argument("--n-phi", type=int, default=40)
    parser.add_argument("--n-z-per-arm", type=int, default=10)
    parser.add_argument("--n-r-cap", type=int, default=5)
    args = parser.parse_args()

    path = generate_center_fed_dipole(
        args.output,
        total_length=args.total_length,
        radius=args.radius,
        feed_gap=args.feed_gap,
        n_phi=args.n_phi,
        n_z_per_arm=args.n_z_per_arm,
        n_r_cap=args.n_r_cap,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
