'Driven two-arm example: qualify direct incident cpu. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import json
import numpy as np
from incident_fields import point_cpu
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
out=Path(__import__('sys').argv[1]);out.mkdir(exist_ok=True)
rng=np.random.default_rng(6018);n=3;nt=8;dx=.01;dt=1e-9;c=299792458.;eps=8.8541878128e-12;coef=dx**3/(4*np.pi*eps);left=-.015
shapes=[(n,n,n),(n+1,n,n),(n,n+1,n),(n,n,n+1)]
positions=[]
for kind,shape in enumerate(shapes):
    idx=np.indices(shape).reshape(3,-1).T;shift=np.full(3,.5)
    if kind:shift[kind-1]=0.
    positions.append(left+(idx+shift)*dx)
hist=[rng.normal(size=(nt,int(np.prod(s))))*1e-7 for s in shapes]
points=np.array([[.026,.012,.007],[-.028,.019,-.014],[.003,.034,.009]])
def vector_reference(point,t,histories):
    E=np.zeros(3);B=np.zeros(3);current=int(np.floor(t/dt+1e-9));oldest=max(current-nt+1,0)
    for kind,(pos,h) in enumerate(zip(positions,histories)):
        rv=point-pos;r=np.linalg.norm(rv,axis=1);tr=t-r/c;tau=tr/dt;lo=np.floor(tau).astype(int);alpha=tau-lo
        hi=lo+1;end=alpha<1e-12;hi[end]=lo[end];lo[end]=np.maximum(lo[end]-1,0);alpha[end]=np.where(hi[end]>0,1.,0.)
        valid=(tr>=0)&(lo>=oldest)&(hi>=oldest)
        overshoot=hi>current;lo[overshoot]=current;hi[overshoot]=current;alpha[overshoot]=0
        src=np.arange(len(r));a=h[lo%nt,src];b=h[hi%nt,src];value=(1-alpha)*a+alpha*b;der=np.where(lo==hi,0,(b-a)/dt)
        if kind==0:E+=np.sum((value/r**3+der/(c*r*r))[valid,None]*rv[valid],axis=0)*coef
        else:
            direction=np.eye(3)[kind-1];E-=direction*np.sum((der/(c*c*r))[valid])*coef
            B+=np.sum(np.cross(direction,rv)[valid]*(value/r**3+der/(c*r*r))[valid,None],axis=0)*coef/(c*c)
    return np.r_[E,B]
rows=[]
for tick in [0,1,6,14]:
    t=tick*dt
    actual=np.array([point_cpu(*p,*hist,n,n,n,dx,dx,dx,left,left,left,nt,dt,t,c,coef) for p in points])
    expected=np.array([vector_reference(p,t,hist) for p in points])
    errors=[float(np.linalg.norm(actual[:,sl]-expected[:,sl])/max(np.linalg.norm(expected[:,sl]),1e-300)) for sl in [slice(0,3),slice(3,6)]]
    if max(errors)>2e-12:raise RuntimeError((tick,errors))
    rows.append(dict(tick=tick,E_relative=errors[0],B_relative=errors[1]))
# Independent Faraday identity for a linearly varying point-current source.
hist2=[np.zeros_like(x) for x in hist];hist2[1][:,0]=(np.arange(nt)*dt)*1e-1
p=np.array([.026,.012,.007]);t=4*dt;dh=1e-7
def field(p,t):return np.asarray(point_cpu(*p,*hist2,n,n,n,dx,dx,dx,left,left,left,nt,dt,t,c,coef))
jac=np.column_stack([(field(p+dh*np.eye(3)[k],t)[:3]-field(p-dh*np.eye(3)[k],t)[:3])/(2*dh) for k in range(3)])
curl=np.array([jac[2,1]-jac[1,2],jac[0,2]-jac[2,0],jac[1,0]-jac[0,1]])
rv=p-positions[1][0]
db=coef/(c*c)*np.cross(np.array([.1,0.,0.]),rv)/np.linalg.norm(rv)**3
faraday=float(np.linalg.norm(curl+db)/np.linalg.norm(db))
if not np.isfinite(faraday) or faraday>1e-6:raise RuntimeError(('Faraday',faraday))
report=dict(status='DIRECT_INCIDENT_CPU_ARITHMETIC_PASSED_NATIVE_CUDA_GATE_REQUIRED',random_history_tests=rows,point_current_Faraday_relative=faraday,
    scope='Independent vector source sum, charge/current terms, zero prehistory, ring wrap and Faraday sign. Not a coupled PIC qualification or proof that corrected RF boundary accuracy passes.')
(out/'direct_incident_CPU_qualification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
