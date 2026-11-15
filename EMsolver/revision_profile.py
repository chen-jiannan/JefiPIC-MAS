"""Opt-in synchronized, disjoint stage timings. No CuPy dependency at import."""
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import csv
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time


def summarize(values):
    a = sorted(values)
    if not a:
        return {"count": 0}
    def pct(p):
        x=(len(a)-1)*p
        i=int(x)
        return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
    return dict(count=len(a),total_s=sum(a),mean_s=statistics.mean(a),
                median_s=statistics.median(a),p05_s=pct(.05),p95_s=pct(.95),
                min_s=a[0],max_s=a[-1])


def environment(cp=None):
    versions={}
    for name in ('numpy','scipy','numba','cupy-cuda12x','cupy-cuda11x','cupy'):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    r=dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
           logical_cpu_count=os.cpu_count(),versions=versions,
           thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS',
           'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','JEFIPIC_FAST_DEPOSITION')})
    if cp is not None:
        d=cp.cuda.Device()
        p=cp.cuda.runtime.getDeviceProperties(d.id)
        name=p['name']
        r['gpu']=dict(id=d.id,name=name.decode() if isinstance(name,bytes) else str(name),
            total_memory_bytes=int(p['totalGlobalMem']),runtime_version=cp.cuda.runtime.runtimeGetVersion(),
            driver_version=cp.cuda.runtime.driverGetVersion())
    return r


class StageProfile:
    def __init__(self,enabled=False,sync=None,warmup_steps=5,clock=None):
        self.enabled=enabled
        self.sync=sync or (lambda:None)
        self.clock=clock or time.perf_counter
        self.warmup_steps=int(warmup_steps)
        self.step=-1
        self.rows=[]
        self.active=False
        self.step_start=None

    def begin_step(self,step):
        if self.enabled:
            self.sync()
            self.step=int(step)
            self.step_start=self.clock()

    def phase(self):
        return 'setup' if self.step<0 else ('warmup' if self.step<self.warmup_steps else 'steady')

    @contextmanager
    def stage(self,name):
        if not self.enabled:
            yield
            return
        if self.active:
            raise RuntimeError('Overlapping timing sections would double count time')
        self.sync()
        t0=self.clock()
        self.active=True
        status='ok'
        try:
            yield
        except BaseException:
            status='exception'
            raise
        finally:
            try:
                self.sync()  # CUDA errors must propagate
                self.rows.append(dict(step=self.step,phase=self.phase(),stage=name,
                                      seconds=self.clock()-t0,status=status))
            finally:
                self.active=False

    def end_step(self):
        if self.enabled:
            if self.step_start is None:
                raise RuntimeError('begin_step was not called')
            self.sync()
            self.rows.append(dict(step=self.step,phase=self.phase(),stage='step_wall_instrumented',
                                  seconds=self.clock()-self.step_start,status='ok'))
            self.step_start=None

    def write(self,directory,metadata):
        if not self.enabled:
            return
        out=Path(directory)
        out.mkdir(parents=True,exist_ok=True)
        groups=defaultdict(list)
        for r in self.rows:
            if r['status']=='ok':
                groups[(r['phase'],r['stage'])].append(r['seconds'])
        summary=[dict(phase=p,stage=s,**summarize(v)) for (p,s),v in sorted(groups.items())]
        payload=dict(metadata=metadata,warmup_steps=self.warmup_steps,
            incomplete_last_step=self.step_start is not None,
            timing_scope='Synchronized wall time; intrusive, not CUDA-event kernel time',
            accounting='Disjoint stages; step_wall_instrumented is separate and must not be added',summary=summary)
        (out/'performance_profile.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
        with (out/'performance_samples.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['step','phase','stage','seconds','status'])
            w.writeheader()
            w.writerows(self.rows)
