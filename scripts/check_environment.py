#!/usr/bin/env python3
import importlib, platform, sys
print("Python:", sys.version.replace("\n"," "))
print("Platform:", platform.platform())
for name in ("numpy","scipy","matplotlib","h5py","numba"):
    try:
        m=importlib.import_module(name); print(f"{name}: {getattr(m,'__version__','available')}")
    except Exception as e: print(f"{name}: NOT AVAILABLE ({e})")
try:
    import cupy as cp
    print("cupy:", cp.__version__)
    print("CUDA devices:", cp.cuda.runtime.getDeviceCount())
except Exception as e:
    print("cupy: NOT AVAILABLE / no working CUDA (CPU reproduction remains supported)")
    print("      ", e)
