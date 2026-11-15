'Driven two-arm example: qualify feedback cpu. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
sys.path.insert(0,str(ROOT));import two_arm_model as ab
from electric_magnetic_sources import magnetic_geometry
from magnetic_feedback import MagneticFeedback,add_feedback_fields
from feedback_reference import make_bridge,grid_solver,kernel_cpu_dispatch

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);f=np.load(ROOT/'validation/em_feedback_fixture.npz');g=ab.frozen_geometry();ga,_=magnetic_geometry(g,3)
    kernel=kernel_cpu_dispatch(SOURCE_ROOT/'EMsolver/cuda_functions.py');rows=[]
    for mode in ('SI','LHQCD'):
        print('CPU arithmetic feedback:',mode,flush=True);b=make_bridge(mode);s=grid_solver(b)
        magnetic=MagneticFeedback(s,ga.ant_src_points,ga.source_weights,b,xp=np,kernel=kernel)
        pos=b.length_si_to_code(g.ant_src_points);w=b.area_si_to_code(g.source_weights);P=[np.zeros((8,g.num_sources)) for _ in range(3)];K=[x.copy() for x in P]
        actual=[];repeated=0.
        for node in range(len(f['electric_P'])):
            s.t=node*s.dt;pp=b.surface_polarization_density_si_to_code(f['electric_P'][node]);kk=b.surface_current_density_si_to_code(f['electric_K'][node])
            for i in range(3):P[i][node%8]=pp[:,i];K[i][node%8]=kk[:,i]
            raw=[np.zeros(s.total_grid_o) for _ in range(6)]
            kernel[1,128](*[pos[:,i] for i in range(3)],w,*P,*K,g.num_sources,0.,0.,0.,0,8,s.dt,s.t,s.dx_s,s.dy_s,s.dz_s,s.total_grid_o,
                s.x_grid_size_o,s.y_grid_size_o,s.z_grid_size_o,s.dx_o,s.dy_o,s.dz_o,s.x_left_boundary_o,s.y_left_boundary_o,s.z_left_boundary_o,*raw,s.c,False)
            e=tuple([x*s.coeff_E for x in raw[:3]]+[x*s.coeff_B for x in raw[3:]])
            m=magnetic.compute(s,f['magnetic_P'][node],f['magnetic_K'][node]);total=add_feedback_fields(e,m)
            if node in (3,9):
                again=magnetic.compute(s,f['magnetic_P'][node],f['magnetic_K'][node])
                repeated=max(repeated,max(float(np.max(abs(a-z))) for a,z in zip(m,again)))
            v=np.stack(total,axis=1);v[:,:3]/=b.E;v[:,3:]/=b.B;actual.append(v)
        actual=np.asarray(actual);expected=f['expected_'+mode];mask=f['scored_nodes']
        errors=[float(np.linalg.norm(actual[mask,...,sl]-expected[mask,...,sl])/np.linalg.norm(expected[mask,...,sl])) for sl in (slice(0,3),slice(3,6))]
        all_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        if max(errors)>2e-11 or all_error>2e-11 or repeated!=0. or not np.array_equal(actual[0],np.zeros_like(actual[0])):
            raise RuntimeError((mode,errors,all_error,repeated))
        s.t+=2*s.dt
        try:magnetic.compute(s,f['magnetic_P'][-1],f['magnetic_K'][-1])
        except RuntimeError:pass
        else:raise RuntimeError('Skipped-node guard failed')
        rows.append(dict(mode=mode,E_relative=errors[0],B_relative=errors[1],all_interval_relative=all_error,idempotent_max_absolute=repeated,
            replay_nodes=len(actual),scored_nodes=mask.tolist(),history_nodes=8,zero_prehistory=True,skipped_node_guard=True))
        np.savez_compressed(out/('CPU_arithmetic_'+mode+'.npz'),actual_EB=actual,expected_EB=expected)
    bound=('magnetic_feedback.py','feedback_reference.py','two_arm_model.py','electric_magnetic_sources.py','actual_geometry.npz','../../EMsolver/cuda_functions.py')
    report=dict(status='NATIVE_KERNEL_ARITHMETIC_CPU_PASSED_CUDA_GATE_REQUIRED',tests=rows,fixture_sha256=hashlib.sha256((ROOT/'validation/em_feedback_fixture.npz').read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in bound},scope=__doc__)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':run(sys.argv[1])
