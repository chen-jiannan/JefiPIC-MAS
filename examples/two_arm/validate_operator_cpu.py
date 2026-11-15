'Driven two-arm example: validate operator cpu. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]

def run_check(m,path):
    import two_arm_model as ab
    f=np.load(ROOT/'validation/operator_fixture.npz');rows=[]
    if len(m.cs)!=5976 or m.ne!=3984 or m.nm!=1992 or m.theta!=.6:raise RuntimeError('Installed operator configuration mismatch')
    for i,node in enumerate(f['original_nodes']):
        p=f['P_prev'][i];k=f['K_prev'][i];q=m.solve(-m.rhs(f['incident_E'][i],f['incident_B'][i])-m.Hp@p-m.Hk@k)
        e,b=ab.field(f['points_m'],m.g,p,k,q);v=np.c_[e,b];expected=f['expected_EB'][i]
        errors=[float(np.linalg.norm(v[:,sl]-expected[:,sl])/np.linalg.norm(expected[:,sl])) for sl in (slice(0,3),slice(3,6))]
        coefficient=float(np.linalg.norm(q-f['K_now'][i])/np.linalg.norm(f['K_now'][i]))
        row=dict(original_node=int(node),E_relative=errors[0],B_relative=errors[1],coefficient_relative=coefficient);rows.append(row)
        if max(errors)>2e-6 or coefficient>.001:raise RuntimeError(('Installed CPU operator does not reproduce qualified states',row))
    report=dict(status='INSTALLED_COMPATIBLE_OPERATOR_CPU_PASSED',tests=rows,magnetic_test_rank=m.magnetic_test_rank,
        fixture_sha256=hashlib.sha256((ROOT/'validation/operator_fixture.npz').read_bytes()).hexdigest(),scope=__doc__)
    Path(path).write_text(json.dumps(report,indent=2),encoding='utf-8');print('PASSED installed compatible operator CPU replay:',rows,flush=True)
    return report

if __name__=='__main__':
    import two_arm_model as ab
    m=ab.model(ab.frozen_geometry());run_check(m,sys.argv[1])
