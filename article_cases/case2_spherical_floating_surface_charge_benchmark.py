"""Run the existing Case2 floating benchmark with a spherical conductor.

This is a complete replacement for:
    article_cases/case2_spherical_floating_surface_charge_benchmark.py

It is intentionally self-contained:
  - no dependency on case2_spherical_geometry.py;
  - supports --mas-source-offset in metres or as a fraction of R, e.g. 0.35R;
  - patches the runtime geometry only for this spherical adapter;
  - does NOT delete or modify the original finite-cylinder model used by other cases.

Why this version is needed:
  Some Case2 benchmark versions do not call make_straight_dipole_stl().  They
  call generate_center_fed_dipole() directly to create floating_connected_cylinder.stl.
  Therefore this adapter patches BOTH make_straight_dipole_stl() and
  generate_center_fed_dipole() before importing the base benchmark.  When this
  spherical adapter is used, both functions write a sphere STL instead.

Expected log lines:
  [sphere-adapter] ACTIVE final self-contained sphere adapter v3-patch-create-dipole-stl
  [sphere-adapter] MAS source offset override: ...
  [sphere-adapter-lock] intercepted cfg.radius reset ...
  [sphere-adapter-check] Writing SPHERE STL via ...

Example:
  cd JefiPIC_MAS
  python article_cases/case2_spherical_floating_surface_charge_benchmark.py \
    --quick \
    --sphere-radius 0.004 \
    --mas-points 192 \
    --mas-source-offset 0.20R \
    --out case2_debug_sphere_offset_0p20R_final
"""

from __future__ import annotations

from pathlib import Path
import argparse
import importlib
import sys
from typing import Any, Optional, Tuple

import numpy as np


# =============================================================================
# Self-contained spherical geometry helpers
# =============================================================================

def _sphere_mask_np(pos: np.ndarray, radius: float) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    rr2 = pos[:, 0] * pos[:, 0] + pos[:, 1] * pos[:, 1] + pos[:, 2] * pos[:, 2]
    return rr2 <= float(radius) ** 2


def _sphere_mask_cp(pos, radius: float, cp_module, bridge=None):
    cp = cp_module
    if bridge is not None and not getattr(bridge, "is_si", False):
        pos_si = pos * bridge.TO_L
    else:
        pos_si = pos
    rr2 = pos_si[:, 0] ** 2 + pos_si[:, 1] ** 2 + pos_si[:, 2] ** 2
    return rr2 <= float(radius) ** 2


def _distance_to_sphere_metal(pos: np.ndarray, radius: float) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    rr = np.sqrt(pos[:, 0] * pos[:, 0] + pos[:, 1] * pos[:, 1] + pos[:, 2] * pos[:, 2])
    return np.maximum(rr - float(radius), 0.0)


def _write_uv_sphere_stl(out_path: Path, radius: float, n_lat: int = 24, n_lon: int = 48) -> Path:
    """Write an ASCII UV-sphere STL centered at the origin.

    The filename is deliberately kept as requested by the base code.  So the
    file may still be named floating_connected_cylinder.stl, but the *content*
    is a sphere when this adapter is active.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    radius = float(radius)
    n_lat = int(n_lat)
    n_lon = int(n_lon)
    if radius <= 0.0:
        raise ValueError("Sphere radius must be positive.")
    if n_lat < 4 or n_lon < 8:
        raise ValueError("Use n_lat >= 4 and n_lon >= 8 for a useful sphere STL.")

    def vertex(i_lat: int, i_lon: int) -> np.ndarray:
        theta = np.pi * i_lat / n_lat
        phi = 2.0 * np.pi * (i_lon % n_lon) / n_lon
        st = np.sin(theta)
        return radius * np.array(
            [st * np.cos(phi), st * np.sin(phi), np.cos(theta)],
            dtype=np.float64,
        )

    def normal_of(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn <= 1.0e-300:
            n = a + b + c
            nn = np.linalg.norm(n)
        if nn <= 1.0e-300:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return n / nn

    triangles = []
    for i in range(n_lat):
        for j in range(n_lon):
            if i == 0:
                triangles.append((vertex(0, j), vertex(1, j), vertex(1, j + 1)))
            elif i == n_lat - 1:
                triangles.append((vertex(i, j), vertex(n_lat, j), vertex(i, j + 1)))
            else:
                a = vertex(i, j)
                b = vertex(i + 1, j)
                c = vertex(i + 1, j + 1)
                d = vertex(i, j + 1)
                triangles.append((a, b, c))
                triangles.append((a, c, d))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("solid sphere_probe_case2\n")
        for a, b, c in triangles:
            n = normal_of(a, b, c)
            centroid = (a + b + c) / 3.0
            if float(np.dot(n, centroid)) < 0.0:
                b, c = c, b
                n = normal_of(a, b, c)

            f.write(f"  facet normal {n[0]:.16e} {n[1]:.16e} {n[2]:.16e}\n")
            f.write("    outer loop\n")
            for p in (a, b, c):
                f.write(f"      vertex {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid sphere_probe_case2\n")

    return out_path


# =============================================================================
# Adapter CLI
# =============================================================================

def _parse_mas_source_offset(value: Optional[str], sphere_radius: float) -> Optional[float]:
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    if s.lower().endswith("r"):
        numeric = s[:-1].strip()
        if not numeric:
            raise ValueError("Use --mas-source-offset 0.35R, not just R.")
        offset = float(numeric) * float(sphere_radius)
    else:
        offset = float(s)

    if not np.isfinite(offset):
        raise ValueError("--mas-source-offset must be finite.")
    if offset <= 0.0:
        raise ValueError("--mas-source-offset must be positive.")
    if offset >= float(sphere_radius):
        raise ValueError("--mas-source-offset must be smaller than --sphere-radius.")
    return float(offset)


def _parse_adapter_args(argv: list[str]) -> Tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sphere-radius", type=float, default=4.0e-3)
    p.add_argument("--sphere-n-lat", type=int, default=24)
    p.add_argument("--sphere-n-lon", type=int, default=48)
    p.add_argument(
        "--mas-source-offset",
        default=None,
        help="Inward source displacement in metres or as a fraction of radius, e.g. 0.35R.",
    )
    p.add_argument(
        "--base-script",
        default="case2_floating_surface_charge_benchmark",
        help="Module name of the existing Case2 dynamic benchmark.",
    )
    p.add_argument("--help-sphere-adapter", action="store_true")
    known, remaining = p.parse_known_args(argv[1:])

    if known.help_sphere_adapter:
        p.print_help()
        raise SystemExit(0)

    return known, [argv[0]] + remaining


# =============================================================================
# Config locking
# =============================================================================

def _patch_BaseConfig_class_setattr(
    BaseConfigClass: type,
    *,
    sphere_radius: float,
    sphere_n_lat: int,
    sphere_n_lon: int,
    mas_source_offset_m: Optional[float],
) -> None:
    """Protect sphere fields from later preset resets inside the base script."""
    if getattr(BaseConfigClass, "_sphere_adapter_final_class_patched", False):
        return

    original_setattr = BaseConfigClass.__setattr__

    def locked_setattr(self, name, value):
        lock_active = bool(getattr(self, "_sphere_adapter_lock_active", False))
        if lock_active:
            forced = None
            if name == "radius":
                forced = float(sphere_radius)
            elif name == "total_length":
                forced = 2.0 * float(sphere_radius)
            elif name == "feed_gap":
                forced = 0.0
            elif name == "n_phi":
                forced = int(sphere_n_lon)
            elif name == "mas_source_offset" and mas_source_offset_m is not None:
                forced = float(mas_source_offset_m)

            if forced is not None:
                try:
                    old_value = getattr(self, name)
                except Exception:
                    old_value = None
                if value != forced:
                    print(
                        "[sphere-adapter-lock] intercepted cfg."
                        f"{name} reset: requested={value!r}, forced={forced!r}, old={old_value!r}"
                    )
                value = forced

        return original_setattr(self, name, value)

    BaseConfigClass.__setattr__ = locked_setattr
    BaseConfigClass._sphere_adapter_final_class_patched = True
    BaseConfigClass._sphere_adapter_final_original_setattr = original_setattr


def _apply_sphere_config(
    cfg: Any,
    *,
    sphere_radius: float,
    sphere_n_lat: int,
    sphere_n_lon: int,
    mas_source_offset_m: Optional[float],
    activate_lock: bool = True,
) -> Any:
    # Temporarily disable lock while applying intended values.
    try:
        object.__setattr__(cfg, "_sphere_adapter_lock_active", False)
    except Exception:
        setattr(cfg, "_sphere_adapter_lock_active", False)

    cfg.case2_geometry = "sphere"
    cfg.sphere_radius = float(sphere_radius)
    cfg.sphere_n_lat = int(sphere_n_lat)
    cfg.sphere_n_lon = int(sphere_n_lon)
    cfg.sphere_adapter_version = "final_self_contained_v3_patch_create_dipole_stl"

    cfg.radius = float(sphere_radius)
    cfg.total_length = 2.0 * float(sphere_radius)
    cfg.feed_gap = 0.0
    cfg.n_phi = int(sphere_n_lon)

    if mas_source_offset_m is not None:
        cfg.mas_source_offset = float(mas_source_offset_m)
        cfg.sphere_mas_source_offset = float(mas_source_offset_m)
        cfg.sphere_mas_source_offset_over_R = float(mas_source_offset_m) / float(sphere_radius)

    if activate_lock:
        try:
            object.__setattr__(cfg, "_sphere_adapter_lock_active", True)
        except Exception:
            setattr(cfg, "_sphere_adapter_lock_active", True)

    return cfg


def _make_patched_BaseConfig(
    OriginalBaseConfig,
    *,
    sphere_radius: float,
    sphere_n_lat: int,
    sphere_n_lon: int,
    mas_source_offset_m: Optional[float],
):
    def PatchedBaseConfig(*args, **kwargs):
        cfg = OriginalBaseConfig(*args, **kwargs)
        _apply_sphere_config(
            cfg,
            sphere_radius=sphere_radius,
            sphere_n_lat=sphere_n_lat,
            sphere_n_lon=sphere_n_lon,
            mas_source_offset_m=mas_source_offset_m,
            activate_lock=True,
        )
        print(
            "[sphere-adapter-check] locked BaseConfig instance: "
            f"radius={cfg.radius:.6e}, total_length={cfg.total_length:.6e}, "
            f"mas_source_offset={getattr(cfg, 'mas_source_offset', None)}"
        )
        return cfg

    return PatchedBaseConfig


# =============================================================================
# Geometry patching
# =============================================================================

def _patch_common_geometry(
    common: Any,
    *,
    sphere_radius: float,
    sphere_n_lat: int,
    sphere_n_lon: int,
) -> dict[str, Any]:
    """Patch common functions before the base benchmark is imported.

    This is the key fix: generate_center_fed_dipole() is also patched, because
    some local Case2 scripts call it directly instead of make_straight_dipole_stl().
    """
    cp = getattr(common, "cp", None)
    get_unit_bridge = getattr(common, "get_unit_bridge", lambda cfg: None)

    def antenna_mask_np(pos, cfg):
        r = float(getattr(cfg, "sphere_radius", getattr(cfg, "radius", sphere_radius)))
        return _sphere_mask_np(pos, r)

    def antenna_mask_cp(pos, cfg, bridge=None):
        if cp is None:
            raise RuntimeError("CuPy is unavailable, but antenna_mask_cp was called.")
        if bridge is None:
            bridge = get_unit_bridge(cfg)
        r = float(getattr(cfg, "sphere_radius", getattr(cfg, "radius", sphere_radius)))
        return _sphere_mask_cp(pos, r, cp_module=cp, bridge=bridge)

    def distance_to_straight_dipole_metal(pos, cfg):
        r = float(getattr(cfg, "sphere_radius", getattr(cfg, "radius", sphere_radius)))
        return _distance_to_sphere_metal(pos, r)

    def antenna_exclusion_mask_np(pos, cfg, clearance_m=0.0):
        clearance_m = max(float(clearance_m), 0.0)
        r = float(getattr(cfg, "sphere_radius", getattr(cfg, "radius", sphere_radius)))
        if clearance_m <= 0.0:
            return _sphere_mask_np(pos, r)
        return _distance_to_sphere_metal(pos, r) <= clearance_m

    def make_straight_dipole_stl(cfg, out_path):
        r = float(getattr(cfg, "sphere_radius", getattr(cfg, "radius", sphere_radius)))
        n_lat = int(getattr(cfg, "sphere_n_lat", sphere_n_lat))
        n_lon = int(getattr(cfg, "sphere_n_lon", sphere_n_lon))
        print(
            "[sphere-adapter-check] Writing SPHERE STL via make_straight_dipole_stl: "
            f"radius={r:.6e}, mesh={n_lat}x{n_lon}, path={out_path}"
        )
        return _write_uv_sphere_stl(Path(out_path), radius=r, n_lat=n_lat, n_lon=n_lon)

    def generate_center_fed_dipole(out_path, *args, **kwargs):
        """Replacement for common.generate_center_fed_dipole when using this adapter.

        The original function writes a finite cylinder.  In this spherical adapter
        we preserve the same signature loosely, but write a UV sphere instead.
        If the base script passes radius=cfg.radius, we use it.  Otherwise we use
        the adapter sphere_radius.
        """
        r = float(kwargs.get("sphere_radius", kwargs.get("radius", sphere_radius)))
        n_lat = int(kwargs.get("sphere_n_lat", sphere_n_lat))
        n_lon = int(kwargs.get("sphere_n_lon", sphere_n_lon))
        print(
            "[sphere-adapter-check] Writing SPHERE STL via generate_center_fed_dipole patch: "
            f"radius={r:.6e}, mesh={n_lat}x{n_lon}, path={out_path}"
        )
        return _write_uv_sphere_stl(Path(out_path), radius=r, n_lat=n_lat, n_lon=n_lon)

    replacements = {
        "antenna_mask_np": antenna_mask_np,
        "antenna_mask_cp": antenna_mask_cp,
        "distance_to_straight_dipole_metal": distance_to_straight_dipole_metal,
        "antenna_exclusion_mask_np": antenna_exclusion_mask_np,
        "make_straight_dipole_stl": make_straight_dipole_stl,
        "generate_center_fed_dipole": generate_center_fed_dipole,
    }

    for name, fn in replacements.items():
        setattr(common, name, fn)

    return replacements


def _patch_base_aliases(base: Any, replacements: dict[str, Any]) -> None:
    for name, fn in replacements.items():
        setattr(base, name, fn)


def _patch_save_config(common: Any) -> None:
    """Augment run_config.json with dynamic sphere metadata when possible."""
    if getattr(common, "_sphere_adapter_final_save_config_patched", False):
        return
    if not hasattr(common, "save_config"):
        return

    original_save_config = common.save_config

    def save_config_with_sphere_attrs(output_dir, cfg, name="run_config.json"):
        result = original_save_config(output_dir, cfg, name=name)
        try:
            import json
            path = Path(output_dir) / name
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                for key in (
                    "case2_geometry",
                    "sphere_radius",
                    "sphere_n_lat",
                    "sphere_n_lon",
                    "sphere_adapter_version",
                    "sphere_mas_source_offset",
                    "sphere_mas_source_offset_over_R",
                ):
                    if hasattr(cfg, key):
                        data[key] = getattr(cfg, key)
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            print("[sphere-adapter-warning] could not augment run_config.json:", exc)
        return result

    common.save_config = save_config_with_sphere_attrs
    common._sphere_adapter_final_save_config_patched = True




def _patch_create_dipole_stl_module(
    *,
    sphere_radius: float,
    sphere_n_lat: int,
    sphere_n_lon: int,
) -> None:
    """Patch the original create_dipole_stl module before importing common/base.

    _jefipic_mas_case_common.py imports:
        from create_dipole_stl import generate_center_fed_dipole

    Some Case2 scripts may also import generate_center_fed_dipole directly from
    create_dipole_stl.  Patching only _jefipic_mas_case_common is therefore not
    sufficient.  This function patches the source module first, so later
    `from create_dipole_stl import generate_center_fed_dipole` receives the
    spherical writer.
    """
    try:
        dipole_mod = importlib.import_module("create_dipole_stl")
    except Exception as exc:
        print("[sphere-adapter-warning] could not import create_dipole_stl for patching:", exc)
        return

    def generate_center_fed_dipole(out_path, *args, **kwargs):
        r = float(kwargs.get("sphere_radius", kwargs.get("radius", sphere_radius)))
        n_lat = int(kwargs.get("sphere_n_lat", sphere_n_lat))
        n_lon = int(kwargs.get("sphere_n_lon", sphere_n_lon))
        print(
            "[sphere-adapter-check] Writing SPHERE STL via "
            "create_dipole_stl.generate_center_fed_dipole patch: "
            f"radius={r:.6e}, mesh={n_lat}x{n_lon}, path={out_path}"
        )
        return _write_uv_sphere_stl(Path(out_path), radius=r, n_lat=n_lat, n_lon=n_lon)

    dipole_mod.generate_center_fed_dipole = generate_center_fed_dipole
    print("[sphere-adapter-check] patched create_dipole_stl.generate_center_fed_dipole")


def _extract_out_name_from_forwarded_argv(argv: list[str]) -> str:
    """Extract the base benchmark --out value from forwarded argv.

    The Case2 benchmark convention used in this project writes to:
        article_cases_output/<out_name>/
    If --out is absent, fall back to "output".
    """
    out_name = "output"
    for i, item in enumerate(argv[1:]):
        if item == "--out" and i + 2 <= len(argv[1:]):
            out_name = argv[1:][i + 1]
            break
        if item.startswith("--out="):
            out_name = item.split("=", 1)[1]
            break
    return str(out_name)


def _prewrite_sphere_stl_for_base_output(
    forwarded_argv: list[str],
    *,
    sphere_radius: float,
    sphere_n_lat: int,
    sphere_n_lon: int,
) -> None:
    """Write sphere STL into the base output directory before base.main().

    This avoids a stale-STL failure mode: if the output directory already
    contains floating_connected_cylinder.stl from a previous failed run, some
    base scripts may reuse it instead of regenerating it.  Prewriting the file
    here makes that path contain a sphere even when the base script skips STL
    generation.
    """
    out_name = _extract_out_name_from_forwarded_argv(forwarded_argv)
    out_dir = Path("article_cases_output") / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("floating_connected_cylinder.stl", "sphere_probe_case2.stl"):
        stl_path = out_dir / filename
        _write_uv_sphere_stl(
            stl_path,
            radius=float(sphere_radius),
            n_lat=int(sphere_n_lat),
            n_lon=int(sphere_n_lon),
        )
        print(
            "[sphere-adapter-check] PREWROTE SPHERE STL before base.main: "
            f"radius={sphere_radius:.6e}, mesh={sphere_n_lat}x{sphere_n_lon}, path={stl_path}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    adapter_args, forwarded_argv = _parse_adapter_args(sys.argv)

    sphere_radius = float(adapter_args.sphere_radius)
    sphere_n_lat = int(adapter_args.sphere_n_lat)
    sphere_n_lon = int(adapter_args.sphere_n_lon)

    if sphere_radius <= 0.0:
        raise ValueError("--sphere-radius must be positive.")
    if sphere_n_lat < 4 or sphere_n_lon < 8:
        raise ValueError("--sphere-n-lat must be >= 4 and --sphere-n-lon must be >= 8.")

    mas_source_offset_m = _parse_mas_source_offset(adapter_args.mas_source_offset, sphere_radius)

    # Forward only the base benchmark arguments to the base parser.
    sys.argv = forwarded_argv

    # Patch the original STL writer module before importing the common/base modules.
    # This is necessary because _jefipic_mas_case_common.py and some base scripts
    # may import generate_center_fed_dipole directly from create_dipole_stl.
    _patch_create_dipole_stl_module(
        sphere_radius=sphere_radius,
        sphere_n_lat=sphere_n_lat,
        sphere_n_lon=sphere_n_lon,
    )

    try:
        common = importlib.import_module("_jefipic_mas_case_common")
    except Exception as exc:
        raise RuntimeError(
            "Could not import _jefipic_mas_case_common. Run this script from the "
            "JefiPIC_MAS repository root, with article_cases on the Python path."
        ) from exc

    # Patch BaseConfig class itself before the base script is imported.
    OriginalCommonBaseConfig = getattr(common, "BaseConfig")
    _patch_BaseConfig_class_setattr(
        OriginalCommonBaseConfig,
        sphere_radius=sphere_radius,
        sphere_n_lat=sphere_n_lat,
        sphere_n_lon=sphere_n_lon,
        mas_source_offset_m=mas_source_offset_m,
    )
    common.BaseConfig = _make_patched_BaseConfig(
        OriginalCommonBaseConfig,
        sphere_radius=sphere_radius,
        sphere_n_lat=sphere_n_lat,
        sphere_n_lon=sphere_n_lon,
        mas_source_offset_m=mas_source_offset_m,
    )

    replacements = _patch_common_geometry(
        common,
        sphere_radius=sphere_radius,
        sphere_n_lat=sphere_n_lat,
        sphere_n_lon=sphere_n_lon,
    )
    _patch_save_config(common)

    try:
        base = importlib.import_module(str(adapter_args.base_script))
    except Exception as exc:
        raise RuntimeError(
            f"Could not import base Case2 benchmark module {adapter_args.base_script!r}. "
            "Use --base-script to point to your local Case2 benchmark module name."
        ) from exc

    # Patch aliases imported directly into the base benchmark module.
    _patch_base_aliases(base, replacements)
    base.BaseConfig = common.BaseConfig
    if hasattr(base, "save_config") and hasattr(common, "save_config"):
        base.save_config = common.save_config

    print(
        "[sphere-adapter] ACTIVE final self-contained sphere adapter v3-patch-create-dipole-stl: "
        f"sphere_radius={sphere_radius:.6e} m, mesh={sphere_n_lat}x{sphere_n_lon}"
    )
    if mas_source_offset_m is None:
        print("[sphere-adapter] MAS source offset: using base default")
    else:
        print(
            "[sphere-adapter] MAS source offset override: "
            f"{mas_source_offset_m:.6e} m ({mas_source_offset_m / sphere_radius:.6g}R)"
        )
    print("[sphere-adapter] Original benchmark module:", adapter_args.base_script)
    print("[sphere-adapter] Forwarded argv:", " ".join(sys.argv[1:]))

    _prewrite_sphere_stl_for_base_output(
        sys.argv,
        sphere_radius=sphere_radius,
        sphere_n_lat=sphere_n_lat,
        sphere_n_lon=sphere_n_lon,
    )

    if not hasattr(base, "main"):
        raise RuntimeError(f"Base module {adapter_args.base_script!r} has no main() function.")
    base.main()


if __name__ == "__main__":
    main()
