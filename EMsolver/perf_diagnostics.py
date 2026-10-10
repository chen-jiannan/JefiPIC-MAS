# EMsolver/perf_diagnostics.py
import time
import functools
from collections import defaultdict


def _try_gpu_sync():
    """
    Synchronize default CuPy stream if CuPy is available.
    This is required for honest GPU timing.
    """
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


class PerfTimer:
    def __init__(self, enable_gpu_sync=True, print_every=10):
        self.enable_gpu_sync = enable_gpu_sync
        self.print_every = int(print_every)
        self.total = defaultdict(float)
        self.count = defaultdict(int)
        self.last = {}
        self.step_id = 0

    def sync(self):
        if self.enable_gpu_sync:
            _try_gpu_sync()

    def tic(self):
        self.sync()
        return time.perf_counter()

    def toc(self, name, t0):
        self.sync()
        dt = time.perf_counter() - t0
        self.total[name] += dt
        self.count[name] += 1
        self.last[name] = dt
        return dt

    def section(self, name):
        return _TimerSection(self, name)

    def next_step(self):
        self.step_id += 1
        if self.print_every > 0 and self.step_id % self.print_every == 0:
            self.report_recent()

    def report_recent(self):
        print(f"\n========== MAS/JefiPIC timing at step {self.step_id} ==========")
        items = sorted(self.last.items(), key=lambda x: -x[1])
        step_total = sum(v for _, v in items)
        for name, dt in items:
            share = 100.0 * dt / step_total if step_total > 0 else 0.0
            print(f"{name:40s} last={dt:10.6f} s   share={share:6.2f}%")
        print(f"{'step_total_measured':40s} last={step_total:10.6f} s")

    def report_total(self):
        print("\n========== MAS/JefiPIC total timing ==========")
        items = sorted(self.total.items(), key=lambda x: -x[1])
        grand = sum(v for _, v in items)
        for name, total in items:
            avg = total / max(self.count[name], 1)
            share = 100.0 * total / grand if grand > 0 else 0.0
            print(
                f"{name:40s} total={total:10.4f} s "
                f"avg={avg:10.6f} s   share={share:6.2f}%"
            )
        print(f"{'grand_total_measured':40s} total={grand:10.4f} s")


class _TimerSection:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = self.timer.tic()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timer.toc(self.name, self.t0)
        return False


def array_backend_name(x):
    mod = type(x).__module__
    if mod.startswith("cupy"):
        return "cupy"
    if mod.startswith("numpy"):
        return "numpy"
    return mod


def print_array_info(name, x):
    """
    Print array backend, shape, dtype, and approximate size.
    Useful to catch accidental GPU->CPU full-field copies.
    """
    try:
        shape = x.shape
        dtype = x.dtype
        size = x.size
        nbytes = x.nbytes
    except Exception:
        print(f"{name}: not an array-like object, type={type(x)}")
        return

    print(
        f"{name:32s} backend={array_backend_name(x):8s} "
        f"shape={shape!s:20s} dtype={str(dtype):10s} "
        f"size={size:.3e} nbytes={nbytes / 1024**2:.2f} MB"
    )


def guard_against_full_grid_cpu_mas(points, grid_points, name="MAS CPU point-source field"):
    """
    Raise if MAS tries to evaluate point-source fields on the full grid on CPU.

    This is the dangerous path:
        compute_point_source_fields(self.grid_points, ...)

    That path can make MAS much slower than the original JefiPIC.
    """
    if points is grid_points:
        raise RuntimeError(
            f"{name} is trying to evaluate fields on full grid_points on CPU. "
            "This is a severe performance bug for production runs. "
            "Use EMsolver.push_antenna_dipole_state(P, K) and GPU Jefimenko recompute instead."
        )

    try:
        if hasattr(points, "shape") and hasattr(grid_points, "shape"):
            if points.shape == grid_points.shape and points.size == grid_points.size:
                raise RuntimeError(
                    f"{name} received an array with the same shape as full grid_points. "
                    "This likely means full-grid CPU MAS fallback was triggered."
                )
    except Exception:
        pass


def timed_method(timer, name):
    """
    Decorator helper if you want to wrap class methods.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with timer.section(name):
                return fn(*args, **kwargs)
        return wrapped
    return deco
