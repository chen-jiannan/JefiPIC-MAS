'PEC sphere and particle-cloud example: validate gpu. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import argparse,hashlib,json,os,sys
import numpy as np
import boundary as b

ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
if not __debug__:
    raise RuntimeError('Qualification requires Python assertions; do not use -O or PYTHONOPTIMIZE.')


def run(out,gpu,fixture_path=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',revision=b.VERSION,native_CUDA=True)
    try:
        # The runner maps the selected physical UUID to one visible device.
        os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(gpu))
        sys.path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
        import cupy as cp
        from numba import config as numba_config
        if numba_config.ENABLE_CUDASIM:
            raise RuntimeError('Native CUDA gate cannot run with NUMBA_ENABLE_CUDASIM=1.')
        from EMsolver.solver import EMsolver
        cp.cuda.Device(0).use()
        fixture=np.load(fixture_path or ROOT/'validation/gpu_replay_fixture.npz');P=fixture['P'];K=fixture['K']
        dt=float(fixture['dt_s']);points=fixture['points_m'];g=b.geometry(.01);hist=8
        rho=cp.zeros(8,dtype=cp.float64)
        faces=tuple(cp.zeros(shape,dtype=cp.float64) for shape in [(3,2,2),(2,3,2),(2,2,3)])
        attrs=dict(cp.cuda.runtime.getDeviceProperties(0))
        name=attrs['name'];report['device']=name.decode() if isinstance(name,bytes) else str(name)
        report['cupy_version']=cp.__version__;report['visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
        def make(point,hermite):
            h=.001
            s=EMsolver(hist,1,1,1,2,2,2,h,h,h,*(point-.5*h),.015,.015,.015,-.015,-.015,.01,dt,b.EPS0,b.C0,0)
            s.set_grid_current_layout('staggered_face');s.set_kernel_mode('atomic')
            assert s.get_configuration()['antenna_time_basis_name']=='legacy'
            if hermite:s.set_antenna_temporal_basis('hermite_P')
            s.set_antenna_sources(g.ant_src_points,g.source_weights,softening_distance=0.)
            s.validate_history_capacity(action='raise')
            return s
        rows=[]
        for route in ['two_pass','direct','legacy_default']:
            solvers=[make(point,route!='legacy_default') for point in points]
            PP=P.copy()
            if route=='legacy_default':
                PP[0]=0
                for n in range(1,len(PP)):PP[n]=PP[n-1]+.5*dt*(K[n-1]+K[n])
            actual=[];expected=[];sampler_error=0.
            for current in range(len(PP)):
                vals=[]
                for point,s in zip(points,solvers):
                    args=dict(antenna_polarization=PP[current],antenna_surface_currents=K[current])
                    if route=='two_pass':
                        s.Jefimenko_solver(rho,face_currents=faces,include_antenna=False)
                        tick=s.time_tick;s.recompute_current_fields(**args)
                        assert s.time_tick==tick
                    else:s.Jefimenko_solver(rho,face_currents=faces,**args)
                    cp.cuda.Device().synchronize()
                    vals.append([float(cp.asnumpy(x)[0]) for x in s.acquire_EB_field()])
                    if current==len(PP)-1:
                        bs=s.sample_current_B_at_points(point[None,:])[0]
                        sampler_error=max(sampler_error,float(np.linalg.norm(bs-vals[-1][3:])/max(np.linalg.norm(bs),1.e-300)))
                actual.append(vals)
                e,bb=b.field_history(points,g,PP,K,current,dt,history_nodes=hist)
                expected.append(np.concatenate((e,bb),axis=1))
            actual=np.array(actual);expected=np.array(expected)
            errors=[float(np.linalg.norm(actual[...,sl]-expected[...,sl])/max(np.linalg.norm(expected[...,sl]),1.e-300)) for sl in [slice(0,3),slice(3,6)]]
            assert np.isfinite(actual).all() and max(errors)<2.e-9,(route,errors)
            assert sampler_error<1.e-12
            if route!='legacy_default':
                try:solvers[0].set_antenna_temporal_basis('legacy')
                except RuntimeError:pass
                else:raise AssertionError('History basis changed after writing nodes.')
            rows.append(dict(route=route,samples=len(PP),history_nodes=hist,E_relative=errors[0],B_relative=errors[1],B_sampler_relative=sampler_error))
            np.savez_compressed(out/(route+'_replay.npz'),actual_EB=actual,CPU_reference_EB=expected)
            print('CUDA replay passed:',rows[-1],flush=True)
        report['routes']=rows;report['status']='NATIVE_CUDA_QUALIFICATION_PASSED'
        report['scope']='Actual original plasma kernels with zero plasma sources, new point history kernel, solver dispatch/scaling, direct/two-pass routes, ring wrap, legacy default, and B sampler. Not full coupled accuracy.'
        return report
    except BaseException as e:
        report['status']='failed';report['error']=repr(e);raise
    finally:(out/'gpu_qualification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--out',required=True);ap.add_argument('--fixture')
    a=ap.parse_args();run(a.out,a.gpu,a.fixture)
