'Driven two-arm example: qualify reflective cic fix. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
from reflecting_sources import deposit_rho,deposit_current
source=SOURCE_ROOT/'Particle_transport/fast_deposition.py'
nodes=[]
for node in ast.parse(source.read_text()).body:
    if isinstance(node,ast.If):
        for item in node.body:
            if isinstance(item,ast.FunctionDef):item.decorator_list=[];nodes.append(item)
old={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),old)
rng=np.random.default_rng(20261114);rows=[]
out=Path(__import__('sys').argv[1]);out.mkdir(exist_ok=True)

def run(name,p0,p1,n=4,scale=1.,charge=1.):
    left=np.full(3,-.09)*scale;spacing=np.full(3,.18/n)*scale
    # Input points are fractions of physical box; physical volume is unchanged.
    p0=(np.asarray(p0)*.18-.09)*scale;p1=(np.asarray(p1)*.18-.09)*scale
    r0=np.zeros((n,n,n));r1=r0.copy();jx=np.zeros((n+1,n,n));jy=np.zeros((n,n+1,n));jz=np.zeros((n,n,n+1));dt=1e-9
    deposit_rho(r0,p0,charge,left,spacing);deposit_rho(r1,p1,charge,left,spacing)
    deposit_current(jx,jy,jz,p0,p1,np.ones(len(p0),bool),charge,dt,left,spacing)
    div=np.diff(jx,axis=0)/spacing[0]+np.diff(jy,axis=1)/spacing[1]+np.diff(jz,axis=2)/spacing[2]
    dr=(r1-r0)/dt;defect=dr+div;den=np.max(abs(dr))+np.max(abs(div))
    rel=float(np.max(abs(defect))/max(den,1e-300))
    mass=max(abs(r0.sum()-len(p0)*charge),abs(r1.sum()-len(p0)*charge))/max(abs(len(p0)*charge),1e-300)
    flux=max(np.max(abs(jx[[0,-1]])),np.max(abs(jy[:,[0,-1]])),np.max(abs(jz[:,:,[0,-1]])))
    if max(rel,mass)>2e-11 or flux!=0:raise RuntimeError((name,rel,mass,flux))
    row=dict(case=name,particles=len(p0),grid=n,coordinate_scale=scale,continuity_linf_relative=rel,charge_relative=mass,outer_normal_flux_absolute=float(flux))
    if name=='interior_matches_original':
        ro=np.zeros_like(r1);o0=np.zeros_like(r0);ox=np.zeros_like(jx);oy=np.zeros_like(jy);oz=np.zeros_like(jz)
        for rr,pp in [(ro,p1),(o0,p0)]:old['_deposit_rho_shape_numba'](rr,pp,charge,1,*left,*(left+n*spacing),*spacing,n,n,n)
        old['_deposit_current_shape_numba'](ox,oy,oz,p0,p1,np.ones(len(p0),bool),charge,1,dt,*left,*spacing,n,n,n)
        row['original_interior_current_relative']=float(np.sqrt(sum(np.linalg.norm(x-y)**2 for x,y in zip([jx,jy,jz],[ox,oy,oz])))/max(np.sqrt(sum(np.linalg.norm(y)**2 for y in [ox,oy,oz])),1e-300))
        if row['original_interior_current_relative']>1e-11 or np.max(abs(ro-r1))>1e-11:raise RuntimeError('Interior changed')
    rows.append(row);print(row,flush=True)

run('old_failing_half_cell',[[.05,.35,.425]],[[.075,.355,.43]])
corners=np.array([[x,y,z] for x in [.000001,.999999] for y in [.000001,.999999] for z in [.000001,.999999]])
run('all_eight_corners',corners,.1+.8*corners)
p0=rng.uniform(.00001,.99999,(120,3));p1=rng.uniform(.00001,.99999,(120,3));run('long_reservoir_reset_trajectories',p0,p1,n=17)
run('LHQCD_scale_long_trajectories',p0,p1,n=17,scale=1e8,charge=1e-12)
p0=rng.uniform(.3,.7,(100,3));p1=np.clip(p0+rng.normal(0,.01,p0.shape),.2,.8);run('interior_matches_original',p0,p1,n=4)
report=dict(status='FOLDED_REFLECTIVE_CIC_CPU_QUALIFIED',adapter_sha256=hashlib.sha256((ROOT/'reflecting_sources.py').read_bytes()).hexdigest(),
    scope='Actual new CPU adapter. Identical full physical box and particle positions; no half-cell guard shrink and no qv deposition. Long slot replacement trajectories retain the existing RF mathematical source contract; this is not a particle-collection/conductor-charge physics validation.',rows=rows)
(out/'reflective_CIC_repair_qualification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
