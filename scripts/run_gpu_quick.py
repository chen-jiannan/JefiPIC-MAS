#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()<1:
        raise RuntimeError("No CUDA device visible")
except Exception as e:
    raise SystemExit(f"GPU quick run requires working CUDA/CuPy: {e}\nSee docs/GPU_SETUP.md")
cmd=[
    sys.executable,
    "article_cases/step4d_a_two_way_retarded_mot_closure_full_em_atomic.py",
    "--quick",
    "--v2e-report",str(ROOT/"reference_data"/"quasi_auxiliary"/"step4c3_v2e_report.json"),
    "--accept-known-c3-limitation",
    "--out","github_full_em_gpu_quick",
]
raise SystemExit(subprocess.call(cmd,cwd=ROOT))
