"""Recompute the augmented MAS side against archived independent SCUFF phasors.

CPU only; this does not rerun SCUFF or advance particles. The same augmented
operator, frozen normalization and theta as the completed RF run are used.
"""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='4'
import sys,json
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'examples/two_arm'))
import two_arm_model as ab

if __name__=='__main__':
 out=HERE/'outputs/scuff_mas';out.mkdir(parents=True,exist_ok=True)
 ref=dict(np.load(HERE/'data/scuff_periodic_fields.npz'))
 g=ab.frozen_geometry();m=ab.model(g);omega=2*np.pi*3e6
 Einc,Binc=ab.reference(g.solve_obs_points)
 print('Solving discrete periodic response of 5976-unknown augmented operator ...',flush=True)
 q,chi,solver=m.periodic_response(m.rhs(Einc,Binc),omega)
 z=np.exp(-1j*omega*1e-9);E,B=m.field(ref['points_m'],z*chi*q,z*q,q)
 rel=lambda x,y:float(np.linalg.norm(x-y)/np.linalg.norm(y))
 result=dict(**solver,E_offset=rel(E[320:640],ref['SCUFF_E'][:320]),
  B_offset=rel(B[320:640],ref['SCUFF_B'][:320]),
  E_probes=rel(E[640:],ref['SCUFF_E'][320:]),B_probes=rel(B[640:],ref['SCUFF_B'][320:]),
  scope='CPU recomputation of the MAS periodic response; archived actual SCUFF reference; no new GPU/PIC/SIE solve')
 expected=json.loads((HERE/'data/scuff_validation.json').read_text())['SCUFF']
 result['maximum_metric_difference_from_archive']=max(abs(result[k]-expected[k]) for k in ['E_offset','B_offset','E_probes','B_probes'])
 assert result['maximum_metric_difference_from_archive']<1e-6
 np.savez_compressed(out/'fields.npz',points_m=ref['points_m'],E=E,B=B)
 (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2),flush=True)
