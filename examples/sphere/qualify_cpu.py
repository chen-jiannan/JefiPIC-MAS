'PEC sphere and particle-cloud example: qualify cpu. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'4')
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
import boundary as b
from EMsolver.mas_mixed_mot import reference_balance
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
if not __debug__:raise RuntimeError('Do not use Python -O for qualification')

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','boundary_revision':b.VERSION,'theta':b.THETA,'ridge_rel':b.RIDGE_REL,'magnetic_weight':b.MAGNETIC_WEIGHT,'native_CUDA_tested':False}
    try:
        cfg=json.loads((ROOT/'input/case.json').read_text())
        assert hashlib.sha256((ROOT/'input/particles.npz').read_bytes()).hexdigest()==cfg['particle_file_sha256']
        preserve=json.loads((ROOT/'source_preservation.json').read_text())
        for name,expected in preserve.items():
            assert hashlib.sha256((SOURCE_ROOT/name).read_bytes()).hexdigest()==expected,name
        report['original_PIC_and_plasma_sources_preserved']=True
        g=b.geometry(cfg['radius_m']);dt=cfg['jefipic_dt_s']
        x=np.load(ROOT/'calibration_incident.npz');E=x['E'];B=x['B']
        assert np.max(abs(x['collocation_points_m']-g.solve_obs_points))<1e-15
        et=np.stack((np.sum(E*g.solve_tangent1,axis=-1),np.sum(E*g.solve_tangent2,axis=-1)),axis=-1)
        beta=reference_balance(g,E,B)
        assert abs(beta/b.MAGNETIC_WEIGHT-1)<1e-12
        hp,hk,prep,delay=b.assemble(g,dt);report['delay']=delay
        spectrum=b.homogeneous_spectrum(hp,hk,prep,dt);report['spectrum']=spectrum
        assert spectrum['modes_above_1_plus_tolerance']==0,spectrum
        rng=np.random.default_rng(305);p=dt*rng.normal(size=384);k=rng.normal(size=384);q=rng.normal(size=384)
        e,bb=b.field_interval(g.solve_obs_points,g,p,k,q,dt)
        direct=b.boundary_coeffs(e,bb,g);assembled=hp@p+hk@k+b.operator_now_coeff(prep,q)
        errors=[float(np.linalg.norm(direct[s]-assembled[s])/np.linalg.norm(direct[s])) for s in (slice(0,1536),slice(1536,None))]
        report['arbitrary_history_operator_E_B_relative']=errors
        assert max(errors)<1e-12,errors
        p=np.zeros(384);k=p.copy();ph=[];kh=[];peak=0.;nums=np.zeros(4)
        for e,bb in zip(E,B):
            rhs=b.boundary_coeffs(e,bb,g);history=hp@p+hk@k
            q=b.solve_fixed_ridge(prep,-rhs-history);pn=p+dt*((1-b.THETA)*k+b.THETA*q)
            res=rhs+history+b.operator_now_coeff(prep,q)
            nums+=np.array([np.linalg.norm(res[:1536])**2,np.linalg.norm(rhs[:1536])**2,np.linalg.norm(res[1536:])**2,np.linalg.norm(rhs[1536:])**2])
            peak=max(peak,float(np.linalg.norm(q)))
            assert np.isfinite(q).all() and np.isfinite(pn).all()
            ph.append(b.coeff_to_vectors(g,pn));kh.append(b.coeff_to_vectors(g,q));p,k=pn,q
        report['frozen_incident_replay']={'samples':len(ph),'finite':True,'max_K_coeff_norm':peak,
            'collocation_Et_relative':float(np.sqrt(nums[0]/nums[1])),'collocation_Bn_relative':float(np.sqrt(nums[2]/nums[3])),
            'collocation_Bn_over_full_incident_B':float(np.sqrt(nums[2])/(b.MAGNETIC_WEIGHT*b.C0*np.linalg.norm(B))),
            'scope':'Boundary response to old incident records only, not a new coupled PIC or WarpX comparison.'}
        assert report['frozen_incident_replay']['collocation_Et_relative']<.05
        assert report['frozen_incident_replay']['collocation_Bn_over_full_incident_B']<.05
        points=np.array(cfg['probes_m'])[[1,2,3]]
        np.savez_compressed(out/'gpu_replay_fixture.npz',P=np.array(ph[:45]),K=np.array(kh[:45]),dt_s=dt,points_m=points)
        report['status']='CPU_QUALIFICATION_PASSED_NATIVE_CUDA_GATE_REQUIRED'
        print(json.dumps(report,indent=2),flush=True)
        return report
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:(out/'cpu_qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=ROOT/'validation');a=ap.parse_args();run(a.out)
