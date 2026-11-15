"""Same-operator full-EM backend timing, NOT end-to-end PIC throughput."""
from pathlib import Path
import argparse,gc,json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from EMsolver.revision_profile import environment,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grids',default='5,9,13,17')
    p.add_argument('--history',type=int,default=16)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--tol',type=float,default=5e-10)
    p.add_argument('--out',default='outputs/field_backends.json')
    a=p.parse_args()
    grids=[int(x) for x in a.grids.split(',')]
    if a.history<4 or min(a.warmup,a.repeats,a.trials)<1 or min(grids)<2:
        p.error('history>=4; grids>=2; warmup/repeats/trials must be positive')
    import numpy as np
    import cupy as cp
    from support.field_test_sources import build_histories,load_and_run,rel_l2
    from EMsolver.solver import EMsolver
    def make_solver(mode,steps,nx,ny,nz,dt,c,eps0):
        dx,dy,dz=.18/nx,.18/ny,.18/nz
        s=EMsolver(steps,nx,ny,nz,nx,ny,nz,dx,dy,dz,0.,0.,0.,
                   dx,dy,dz,0.,0.,0.,dt,eps0,c,False,"open")
        s.set_grid_current_layout("staggered_face")
        s.set_kernel_mode(mode)
        return s
    result=dict(environment=environment(cp),arguments=vars(a),cases=[],
        scope='Full _run_jefimenko_kernel call: allocation, charge/current kernels, scaling; no PIC or MAS',
        memory_scope='CuPy pool snapshots, NOT peak device memory')
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    def save():
        out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for n in grids:
        rho,faces=build_histories(a.history,n,n,n)
        case=dict(grid_n=n,n_observations=n**3,n_charge_sources=n**3,
                  n_current_faces=3*(n+1)*n*n,trials=[])
        result['cases'].append(case)
        for trial in range(a.trials):
            order=['atomic','observation'] if trial%2==0 else ['observation','atomic']
            fields={}
            row=dict(trial=trial,order=order,backends={})
            for mode in order:
                s=make_solver(mode,a.history,n,n,n,1e-9,299792458.,8.8541878128e-12)
                config=s.get_configuration()
                if config['kernel_mode']!=mode or config['is_electrostatic']:
                    raise RuntimeError('wrong physical model or backend')
                t0=time.perf_counter()
                load_and_run(s,rho,faces)
                cold=time.perf_counter()-t0
                for _ in range(a.warmup):
                    s._run_jefimenko_kernel(include_antenna=False)
                cp.cuda.Device().synchronize()
                samples=[]
                for _ in range(a.repeats):
                    cp.cuda.Device().synchronize()
                    t0=time.perf_counter()
                    s._run_jefimenko_kernel(include_antenna=False)
                    cp.cuda.Device().synchronize()
                    samples.append(time.perf_counter()-t0)
                fields[mode]=(np.stack([cp.asnumpy(x) for x in (s.GEx,s.GEy,s.GEz)],axis=1),
                              np.stack([cp.asnumpy(x) for x in (s.GBx,s.GBy,s.GBz)],axis=1))
                row['backends'][mode]=dict(cold_upload_run_download_s=cold,
                    steady=summarize(samples),samples_s=samples,configuration=config,
                    pool_used_bytes_snapshot=cp.get_default_memory_pool().used_bytes(),
                    pool_reserved_bytes_snapshot=cp.get_default_memory_pool().total_bytes())
                del s
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
            ea,ba=fields['atomic']; eo,bo=fields['observation']
            finite=all(np.isfinite(x).all() for x in (ea,ba,eo,bo))
            er,br=rel_l2(ea,eo),rel_l2(ba,bo)
            row.update(E_rel_l2=er,B_rel_l2=br,finite=finite,
                       parity_pass=bool(finite and er<=a.tol and br<=a.tol))
            row['observation_over_atomic_median_time']=(row['backends']['observation']['steady']['median_s']/
                                                       row['backends']['atomic']['steady']['median_s'])
            case['trials'].append(row)
            save()
            print(f'grid={n} trial={trial} E_rel={er:.3e} B_rel={br:.3e}',flush=True)
            if not row['parity_pass']:
                raise RuntimeError('Backend parity failed; timings unsuitable for performance claims')
    result['status']='complete'
    save()

if __name__=='__main__':
    main()
