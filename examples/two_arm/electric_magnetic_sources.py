'Driven two-arm example: electric magnetic sources. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from types import SimpleNamespace
import numpy as np
from scipy.linalg import cho_factor
from EMsolver.mas_mixed_mot import NormalizedMixedMOT,field_blocks,C0

def magnetic_geometry(g,stride=10):
    ids=np.arange(0,g.num_sources,stride)
    weights=g.source_weights[ids].copy();weights*=g.source_weights.sum()/weights.sum()
    z=np.zeros((len(ids),3));x=z.copy();x[:,0]=1;y=z.copy();y[:,1]=1;zz=z.copy();zz[:,2]=1
    a=SimpleNamespace(ant_src_points=g.ant_src_points[ids],source_weights=weights,source_tangent1=x,source_tangent2=y)
    b=SimpleNamespace(ant_src_points=a.ant_src_points,source_weights=weights,source_tangent1=zz,source_tangent2=z)
    return a,b

def magnetic_maps(points,ga,gb,dt,theta):
    ex,bx=field_blocks(points,ga,dt,chunk=8,theta=theta);ez,bz=field_blocks(points,gb,dt,chunk=8,theta=theta)
    ee=[];bb=[]
    for e1,b1,e2,b2 in zip(ex,bx,ez,bz):
        e=np.concatenate((e1.reshape(len(points)*3,-1,2),e2[:,::2,None]),axis=2).reshape(len(points)*3,-1)
        b=np.concatenate((b1.reshape(len(points)*3,-1,2),b2[:,::2,None]),axis=2).reshape(len(points)*3,-1)
        ee.append(-C0*b);bb.append(e/C0)
    return ee,bb

class Augmented(NormalizedMixedMOT):
    def __init__(self,g,dt,beta,theta=.5,stride=10,ridge=1e-4,factor=True):
        self.g=g;self.dt=dt;self.theta=theta;self.magnetic_weight=beta;self.ridge_rel=ridge
        self.ga,self.gb=magnetic_geometry(g,stride);self.ne=2*g.num_sources;self.nm=3*len(self.ga.ant_src_points)
        m=g.num_collocation_points;n=self.ne+self.nm
        self.Hp=np.empty((3*m,n));self.Hk=np.empty_like(self.Hp);now=np.empty_like(self.Hp)
        for i in range(0,m,8):
            end=min(i+8,m);x=g.solve_obs_points[i:end]
            E,B=field_blocks(x,g,dt,chunk=8,theta=theta);ME,MB=magnetic_maps(x,self.ga,self.gb,dt,theta)
            tangent=np.stack([g.solve_tangent1[i:end],g.solve_tangent2[i:end]],axis=1)
            for target,e,b,me,mb in zip([self.Hp,self.Hk,now],E,B,ME,MB):
                ee=np.c_[e,me].reshape(end-i,3,n);bb=np.c_[b,mb].reshape(end-i,3,n)
                target[2*i:2*end]=np.einsum('mua,man->mun',tangent,ee).reshape(-1,n)
                target[2*m+i:2*m+end]=beta*C0*np.einsum('ma,man->mn',g.solve_normals[i:end],bb)
        area=np.sqrt(g.solve_weights/np.mean(g.solve_weights));self.rs=np.r_[np.repeat(area,2),area]
        now*=self.rs[:,None];self.cs=np.linalg.norm(now,axis=0);now/=self.cs;self.As=now
        rng=np.random.default_rng(1234);v=rng.normal(size=n);v/=np.linalg.norm(v)
        for _ in range(8):v=now.T@(now@v);v/=np.linalg.norm(v)
        self.ridge_alpha=ridge*np.linalg.norm(now@v)
        if factor:
            mat=now.T@now;mat.flat[::n+1]+=self.ridge_alpha**2;self.cf=cho_factor(mat,lower=True,overwrite_a=True,check_finite=False)

    def field(self,points,p,k,q):
        E=[];B=[]
        for i in range(0,len(points),16):
            x=points[i:i+16];e,b=field_blocks(x,self.g,self.dt,chunk=8,theta=self.theta)
            me,mb=magnetic_maps(x,self.ga,self.gb,self.dt,self.theta)
            E.append(sum(ee@v[:self.ne]+mm@v[self.ne:] for ee,mm,v in zip(e,me,[p,k,q])).reshape(-1,3))
            B.append(sum(bb@v[:self.ne]+mm@v[self.ne:] for bb,mm,v in zip(b,mb,[p,k,q])).reshape(-1,3))
        return np.concatenate(E),np.concatenate(B)
