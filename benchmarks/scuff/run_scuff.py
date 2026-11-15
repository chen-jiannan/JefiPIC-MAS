"""Run the matched SCUFF-EM PEC scattering calculation on Linux/Unix.

The script does not install software or alter an existing SCUFF-EM build.
It always writes a return archive, including when a level fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile


HERE = pathlib.Path(__file__).resolve().parent
OMEGA = 6.287535065855045e-05
SOURCE_STRENGTH = ("1", "-0.45", "0.75")
SOURCE_LOCATION = ("4", "3", "12.6")


def sha256(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def archive(run_dir):
    zpath = run_dir.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(run_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(run_dir.parent))
    return zpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=None, help="Path to scuff-scatter; otherwise search PATH")
    ap.add_argument("--levels", nargs="+", choices=("coarse", "medium", "fine"), default=("coarse", "medium"))
    ap.add_argument("--timeout-hours", type=float, default=6.0)
    args = ap.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = HERE / ("scuff_return_" + stamp)
    run_dir.mkdir()
    binary = pathlib.Path(args.binary).resolve() if args.binary else shutil.which("scuff-scatter")
    meta = {
        "status": "started",
        "binary": str(binary) if binary else None,
        "levels": list(args.levels),
        "omega_numeric": OMEGA,
        "source_strength": SOURCE_STRENGTH,
        "source_location_mm_numeric": SOURCE_LOCATION,
        "JefiPIC_time_convention": "exp(+i omega t)",
        "SCUFF_time_convention": "exp(-i omega t)",
        "results": {},
    }
    try:
        if not binary or not pathlib.Path(binary).is_file():
            raise RuntimeError("scuff-scatter not found; install SCUFF-EM or pass --binary /absolute/path/to/scuff-scatter")
        meta["binary_sha256"] = sha256(binary)
        for level in args.levels:
            level_dir = run_dir / level
            level_dir.mkdir()
            names = [
                f"twoarm_{level}.scuffgeo",
                f"lower_{level}.msh",
                f"upper_{level}.msh",
                "scuff_evaluation_points_mm.txt",
            ]
            for name in names:
                shutil.copy2(HERE / name, level_dir / name)
            cmd = [
                str(binary), "--geometry", f"twoarm_{level}.scuffgeo",
                "--Omega", format(OMEGA, ".17g"),
                "--psStrength", *SOURCE_STRENGTH,
                "--psLocation", *SOURCE_LOCATION,
                "--EPFile", "scuff_evaluation_points_mm.txt",
            ]
            with (level_dir / "run.log").open("w", encoding="utf-8") as log:
                log.write("COMMAND: " + " ".join(cmd) + "\n")
                log.flush()
                completed = subprocess.run(
                    cmd, cwd=level_dir, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(1.0, args.timeout_hours * 3600), check=False,
                )
            outputs = [p.name for p in level_dir.iterdir() if p.is_file()]
            meta["results"][level] = {
                "returncode": completed.returncode,
                "files": sorted(outputs),
                "input_sha256": {n: sha256(level_dir / n) for n in names},
            }
            (run_dir / "status.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"{level} scuff-scatter exited {completed.returncode}; see {level_dir / 'run.log'}")
            if not any(p.suffix == ".scattered" for p in level_dir.iterdir()):
                raise RuntimeError(f"{level} produced no .scattered file; see run.log")
        meta["status"] = "SCUFF_SCATTER_LEVELS_COMPLETE_ANALYSIS_PENDING"
    except Exception as exc:
        meta["status"] = "STOPPED_WITH_ERROR"
        meta["error"] = str(exc)
    finally:
        (run_dir / "status.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        zpath = archive(run_dir)
        print(json.dumps(meta, indent=2))
        print("RETURN THIS ZIP:", zpath)
    return 0 if meta["status"] == "SCUFF_SCATTER_LEVELS_COMPLETE_ANALYSIS_PENDING" else 1


if __name__ == "__main__":
    sys.exit(main())
