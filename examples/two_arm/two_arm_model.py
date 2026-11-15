'Driven two-arm example: two arm model. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
sys.path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
import retarded_history as c3
import plasma_setup as record
import drive_and_surface_diagnostics as old
from EMsolver.mas_mixed_mot import NormalizedMixedMOT,C0,EPS0

# The compatible electric/magnetic candidate passed long CPU gates at theta=.6.
# This opt-in package does not alter the separately qualified sphere implementation.
THETA=.6
VERSION='article-compatible-EM-hermite-native-v5-20261115'
FREQUENCIES=np.array([3.,6.,9.,15.])*1e6

def settings(smoke=False):
    s=record.mode_settings(False)
    s.update(cycles=12.,particles_per_species=24000,grid_n=17,seed=101,sources_per_arm=996,base_sources_per_arm=900,feed_cap_extra_sources_per_arm=96,magnetic_sources_per_arm=332)
    if smoke:s.update(cycles=.009,particles_per_species=200,grid_n=7,diag_stride=1,analysis_start_cycles=0.)
    return s

def reference(points,frequency=3e6):
    """Fixed physical dipole used in the qualified SCUFF case; not an RF feed."""
    v=np.asarray(points)-np.array([.004,.003,.0126]);r=np.linalg.norm(v,axis=1);n=v/r[:,None]
    p=np.array([1.,-.45,.75])*1e-12;w=2*np.pi*frequency;k=w/C0
    proj=np.sum(n*p,axis=1)[:,None]
    e=(3*n*proj-p)*(1/r[:,None]**3+1j*k/r[:,None]**2)
    e-=k*k*(n*proj-p)/r[:,None]
    e*=np.exp(-1j*k*r[:,None])/(4*np.pi*EPS0)
    b=np.cross(np.broadcast_to(p,n.shape),n)*(1j*w/r[:,None]**2-w*w/(C0*r[:,None]))
    b*=np.exp(-1j*k*r[:,None])/(4*np.pi*EPS0*C0*C0)
    return e,b

def model(g,magnetic_weight=None):
    from mixed_boundary import CompatibleMixedMOT
    if magnetic_weight is None:magnetic_weight=float(json.loads((ROOT/'calibration.json').read_text())['beta'])
    return CompatibleMixedMOT(g,c3.DT,magnetic_weight,theta=THETA,stride=3,ridge=1e-4)


def now_coeff(s,q):return (s.As@(q*s.cs))/s.rs

def vectors(g,q):
    # The historical time-domain helper coerces float64; harmonic diagnostics
    # require complex coefficients to survive this projection unchanged.
    q=np.asarray(q).reshape(-1,2)
    return q[:,:1]*g.source_tangent1+q[:,1:]*g.source_tangent2

def electric_field(points,g,p,k,q,chunk=32):
    """Apply independent vector dipole formulas without storing dense maps."""
    points=np.asarray(points);E=[];B=[]
    pp=vectors(g,p);kk=vectors(g,k);qq=vectors(g,q)
    for first in range(0,len(points),chunk):
        v=points[first:first+chunk,None,:]-g.ant_src_points[None,:,:]
        r=np.linalg.norm(v,axis=-1);a=1-r/(C0*c3.DT)
        if np.any(a<0) or np.any(a>1) or np.any(r<=0):raise ValueError('Invalid one-step observation delay')
        n=v/r[...,None];a=a[...,None]
        f0=(2*THETA-1)*a**3+(1-3*THETA)*a**2+a
        f1=(1-2*THETA)*a**3+(3*THETA-1)*a**2
        d0=3*(2*THETA-1)*a*a+2*(1-3*THETA)*a+1
        dd0=6*(2*THETA-1)*a+2*(1-3*THETA)
        P=pp+c3.DT*(f0*kk+f1*qq);K=d0*kk+(1-d0)*qq;D=dd0*(kk-qq)/c3.DT
        def T(x):return 3*n*np.sum(n*x,axis=-1)[...,None]-x
        r=r[...,None];weight=g.source_weights[None,:,None]/(4*np.pi*EPS0)
        ee=weight*(T(P)/r**3+T(K)/(C0*r*r)+(n*np.sum(n*D,axis=-1)[...,None]-D)/(C0*C0*r))
        bb=weight/(C0*C0)*(np.cross(K,n)/r**2+np.cross(D,n)/(C0*r))
        E.append(np.sum(ee,axis=1));B.append(np.sum(bb,axis=1))
    return np.concatenate(E),np.concatenate(B)

class HarmonicAccumulator:
    """Online least squares with DC/trend and four fixed frequencies; no field-history dump."""
    def __init__(self,start):self.start=start;self.gram=np.zeros((10,10));self.products={};self.count=0
    def add(self,t,**fields):
        if t<self.start:return
        a=2*np.pi*FREQUENCIES*t
        x=np.r_[1.,(t-self.start)*3e6,np.cos(a),np.sin(a)]
        self.gram+=np.outer(x,x);self.count+=1
        for name,value in fields.items():
            if name not in self.products:self.products[name]=np.zeros((10,)+value.shape)
            self.products[name]+=x.reshape((10,)+(1,)*value.ndim)*value
    def save(self,path):
        if self.count<12:return {'samples':self.count,'status':'insufficient_samples_smoke_only'}
        data={};report={'samples':self.count,'fit_start_s':self.start,'frequencies_Hz':FREQUENCIES.tolist(),'normal_matrix_condition':float(np.linalg.cond(self.gram))}
        for name,p in self.products.items():
            v=np.linalg.solve(self.gram,p.reshape(10,-1)).reshape(p.shape)
            data[name]=v[2:6]-1j*v[6:10]
        np.savez_compressed(path,frequencies_Hz=FREQUENCIES,**data)
        return report

def frozen_geometry():
    from types import SimpleNamespace
    data=np.load(ROOT/'actual_geometry.npz')
    g=SimpleNamespace(**{k:data[k] for k in data.files if not k.startswith('holdout')})
    g.num_sources=len(g.ant_src_points);g.num_points=g.num_sources;g.num_collocation_points=len(g.solve_obs_points)
    return g

def frozen_holdout():
    a=np.load(ROOT/'actual_geometry.npz')
    return dict(points=a['holdout_points'],normals=a['holdout_normals'],mask_matrix=a['holdout_mask_matrix'],region_names=list(a['holdout_region_names']))

def field(points,g,p,k,q,chunk=32):
    from electric_magnetic_sources import magnetic_geometry
    ne=2*g.num_sources;ga,gb=magnetic_geometry(g,3)
    def split(v):
        v=np.asarray(v)[ne:].reshape(-1,3)
        return v[:,:2].ravel(),np.c_[v[:,2],np.zeros(len(v))].ravel()
    pa,pz=split(p);ka,kz=split(k);qa,qz=split(q)
    ee,be=electric_field(points,g,p[:ne],k[:ne],q[:ne],chunk)
    ea,ba=electric_field(points,ga,pa,ka,qa,chunk);ez,bz=electric_field(points,gb,pz,kz,qz,chunk)
    return ee-C0*(ba+bz),be+(ea+ez)/C0

def state_norm(m,k,p):
    weights=m.ga.source_weights
    def mag(v):
        vv=np.asarray(v)[m.ne:].reshape(-1,3)
        return float(np.sum(weights[:,None]*abs(vv)**2)/np.sum(weights))
    return float(np.sqrt(.5*(c3.weighted_coeff_rms(m.g,k[:m.ne])**2+c3.weighted_coeff_rms(m.g,p[:m.ne]/m.dt)**2+mag(k)+mag(p/m.dt))))

def collocation_metrics(m,incident_E,incident_B,residual):
    n=2*m.g.num_collocation_points
    area=np.maximum(m.physical_rs[:n],1e-300)
    e=residual[:n]/area
    inc=np.stack((np.sum(incident_E*m.g.solve_tangent1,axis=1),np.sum(incident_E*m.g.solve_tangent2,axis=1)),axis=1).ravel()
    # This is a projected internal solver diagnostic, NEVER a full Bn gate.
    bn=residual[n:]/(m.magnetic_weight*C0)
    bden=np.linalg.norm(incident_B*m.physical_rs[n:,None])
    return e,inc,bn,float(bden)
