'Driven two-arm example: validate feedback gpu. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import argparse,hashlib,json,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]

BOUND_FILES=('magnetic_feedback.py','two_arm_model.py','electric_magnetic_sources.py','mixed_boundary.py','run_simulation.py','feedback_reference.py','validate_operator_cpu.py','actual_geometry.npz','calibration.json','../../EMsolver/cuda_functions.py','validation/em_feedback_fixture.npz','validation/operator_fixture.npz')
def fingerprints():return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in BOUND_FILES}

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);report=dict(status='RUNNING',scope=__doc__,source_sha256=fingerprints())
    try:
        import two_arm_model as ab
        from electric_magnetic_sources import magnetic_geometry
        from magnetic_feedback import MagneticFeedback,observation_points,add_feedback_fields
        from feedback_reference import make_bridge,grid_solver
        import cupy as cp
        from numba import config
        from EMsolver.solver import EMsolver
        from _case_common import JefiPICUnitBridge
        if config.ENABLE_CUDASIM:raise RuntimeError('Native CUDA required; CUDASIM is not accepted')
        cp.cuda.Device(0).use();g=ab.frozen_geometry();ga,_=magnetic_geometry(g,3);f=np.load(ROOT/'validation/em_feedback_fixture.npz');rows=[]
        for mode in ('SI','LHQCD'):
            b=JefiPICUnitBridge(mode=mode);ref=make_bridge(mode);template=grid_solver(ref)
            for attr in ('L','T','Q','I','E','B','c_code','epsilon0_code'):
                if abs(getattr(b,attr)/getattr(ref,attr)-1)>1e-12:raise RuntimeError('Unit bridge differs from CPU reference: '+attr)
            s=EMsolver(8,3,3,3,2,2,2,template.dx_o,template.dy_o,template.dz_o,
                template.x_left_boundary_o,template.y_left_boundary_o,template.z_left_boundary_o,
                template.dx_s,template.dy_s,template.dz_s,-.01*b.L,-.01*b.L,-.01*b.L,template.dt,b.epsilon0_code,b.c_code,False)
            s.set_grid_current_layout('staggered_face');s.set_kernel_mode('atomic');s.set_antenna_temporal_basis('hermite_P')
            s.set_antenna_sources(b.length_si_to_code(g.ant_src_points),b.area_si_to_code(g.source_weights),softening_distance=0.)
            s.validate_history_capacity(action='raise')
            if not np.allclose(observation_points(s)/b.L,f['points_m'],rtol=1e-13,atol=1e-16):raise RuntimeError('Replay receiver mismatch')
            magnetic=MagneticFeedback(s,ga.ant_src_points,ga.source_weights,b)
            rho=cp.zeros(8,dtype=cp.float64);faces=tuple(cp.zeros(shape,dtype=cp.float64) for shape in ((3,2,2),(2,3,2),(2,2,3)))
            actual=[];duplicate_error=0.
            for node in range(len(f['electric_P'])):
                s.Jefimenko_solver(rho,face_currents=faces,include_antenna=False);tick=s.time_tick;t=s.t
                s.recompute_current_fields(antenna_polarization=b.surface_polarization_density_si_to_code(f['electric_P'][node]),
                    antenna_surface_currents=b.surface_current_density_si_to_code(f['electric_K'][node]),include_antenna=True)
                extra=magnetic.compute(s,f['magnetic_P'][node],f['magnetic_K'][node])
                if s.time_tick!=tick or s.t!=t:raise RuntimeError('Feedback changed history time')
                total=add_feedback_fields(s.acquire_EB_field(),extra)
                if node==9:
                    repeat=magnetic.compute(s,f['magnetic_P'][node],f['magnetic_K'][node])
                    duplicate_error=max(float(cp.max(cp.abs(x-y)).get()) for x,y in zip(extra,repeat))
                cp.cuda.Device().synchronize();v=np.stack([cp.asnumpy(x) for x in total],axis=1)
                v[:,:3]/=b.E;v[:,3:]/=b.B;actual.append(v)
            actual=np.asarray(actual);expected=f['expected_'+mode];mask=f['scored_nodes'];errors=[];bridge=[]
            for sl in (slice(0,3),slice(3,6)):
                errors.append(float(np.linalg.norm(actual[mask,...,sl]-expected[mask,...,sl])/np.linalg.norm(expected[mask,...,sl])))
                bridge.append(float(np.linalg.norm(expected[mask,...,sl]-f['expected_SI'][mask,...,sl])/np.linalg.norm(f['expected_SI'][mask,...,sl])))
            np.savez_compressed(out/('native_feedback_'+mode+'.npz'),actual_EB=actual,CPU_reference_EB=expected,scored_nodes=mask)
            if not np.isfinite(actual).all() or max(errors)>2e-8 or max(bridge)>.001 or duplicate_error!=0.:
                raise RuntimeError((mode,'Native electric + magnetic feedback failed',errors,bridge,duplicate_error))
            if not np.array_equal(actual[0],np.zeros_like(actual[0])):raise RuntimeError('Zero-prehistory gate failed')
            rows.append(dict(unit_mode=mode,E_relative=errors[0],B_relative=errors[1],original_unit_bridge_E_relative=bridge[0],original_unit_bridge_B_relative=bridge[1],
                nodes=len(actual),scored_nodes=mask.tolist(),electric_sources=g.num_sources,magnetic_sources=magnetic.count,receivers=s.total_grid_o,
                history_nodes=8,zero_prehistory=True,idempotent_max_absolute=duplicate_error))
            print('PASSED native electric + magnetic feedback:',rows[-1],flush=True)
        props=cp.cuda.runtime.getDeviceProperties(0);name=props['name'];report.update(status='NATIVE_EM_FEEDBACK_PASSED',tests=rows,
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=name.decode() if isinstance(name,bytes) else str(name),native_CUDA=True)
    except BaseException as exc:report.update(status='FAILED',error=repr(exc));raise
    finally:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);run(ap.parse_args().out)
