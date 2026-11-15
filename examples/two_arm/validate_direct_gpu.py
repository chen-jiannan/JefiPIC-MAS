'Driven two-arm example: validate direct gpu. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
sys.path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
from incident_fields import sample_solver,point_cpu

def run(out):
    import cupy as cp
    from numba import config
    from EMsolver.solver import EMsolver
    if config.ENABLE_CUDASIM:raise RuntimeError('Native CUDA required; CUDASIM is not a qualification')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];rng=np.random.default_rng(6018)
    n=3;nt=8;shapes=[(n,n,n),(n+1,n,n),(n,n+1,n),(n,n,n+1)]
    base=[rng.normal(size=(15,)+shape)*1e-7 for shape in shapes]
    points=np.array([[.026,.012,.007],[-.028,.019,-.014],[.003,.034,.009]])
    for name,c,eps,dt in [('SI',299792458.,8.8541878128e-12,1e-9),('code_units',3.,9.063122742868516e13,.1)]:
        dx=.01;left=-.015
        s=EMsolver(nt,n,n,n,n,n,n,dx,dx,dx,left,left,left,dx,dx,dx,left,left,left,dt,eps,c,0)
        s.set_grid_current_layout('staggered_face');s.set_kernel_mode('atomic')
        history=[np.zeros((nt,int(np.prod(shape)))) for shape in shapes];AA=[];EE=[]
        for tick in range(15):
            fields=[b[tick] for b in base]
            s.Jefimenko_solver(cp.asarray(fields[0].ravel()),face_currents=tuple(cp.asarray(x) for x in fields[1:]),include_antenna=False)
            for h,x in zip(history,fields):h[tick%nt]=x.ravel()
            e,b=sample_solver(s,points);actual=np.c_[e,b]
            expected=np.array([point_cpu(*p,*history,n,n,n,dx,dx,dx,left,left,left,nt,dt,s.t,c,s.coeff_E) for p in points])
            AA.append(actual);EE.append(expected)
        actual=np.array(AA);expected=np.array(EE)
        errors=[float(np.linalg.norm(actual[:,:,sl]-expected[:,:,sl])/max(np.linalg.norm(expected[:,:,sl]),1e-300)) for sl in [slice(0,3),slice(3,6)]]
        if not np.isfinite(actual).all() or max(errors)>2e-9:raise RuntimeError((name,errors))
        rows.append(dict(units=name,E_relative=errors[0],B_relative=errors[1],nodes=15,history_nodes=nt))
        np.savez_compressed(out/(name+'_direct.npz'),GPU=actual,CPU=expected)
    report=dict(status='NATIVE_DIRECT_INCIDENT_GPU_QUALIFIED',rows=rows,scope='Actual EMsolver source upload/time dispatch, arbitrary PEC receiver kernel, source ring wrap and code-unit prefactors. Not the coupled experiment.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();run(a.out)
