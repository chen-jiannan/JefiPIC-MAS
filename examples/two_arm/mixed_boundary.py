'Driven two-arm example: mixed boundary. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
import numpy as np
from scipy.linalg import svd,cho_factor
from electric_magnetic_sources import Augmented

class CompatibleMixedMOT(Augmented):
    def __init__(self,g,dt,beta,theta=.6,stride=3,ridge=1e-4):
        super().__init__(g,dt,beta,theta=theta,stride=stride,ridge=ridge,factor=False)
        self.physical_split=2*g.num_collocation_points;self.physical_rs=self.rs.copy()
        B=self.Hp[self.physical_split:,self.ne:]*self.rs[self.physical_split:,None]
        cs=np.linalg.norm(B,axis=0);B/=np.where(cs>1e-300,cs,1.)
        u,s,_=svd(B,full_matrices=False,check_finite=False);keep=s>s[0]*ridge;self.Q=u[:,keep]
        self.magnetic_test_rank=int(keep.sum());self.magnetic_test_cutoff=float(s[0]*ridge)
        def compress(weighted):return np.concatenate((weighted[:self.physical_split],self.Q.T@weighted[self.physical_split:]),axis=0)
        self.Hp=compress(self.Hp*self.physical_rs[:,None]);self.Hk=compress(self.Hk*self.physical_rs[:,None])
        now=compress(self.As*self.cs);self.rs=np.ones(len(now));self.cs=np.linalg.norm(now,axis=0);self.cs=np.where(self.cs>1e-300,self.cs,1.);self.As=now/self.cs
        n=len(self.cs);v=np.random.default_rng(1234).normal(size=n);v/=np.linalg.norm(v)
        for _ in range(10):v=self.As.T@(self.As@v);v/=np.linalg.norm(v)
        self.ridge_alpha=ridge*np.linalg.norm(self.As@v);matrix=self.As.T@self.As;matrix.flat[::n+1]+=self.ridge_alpha**2
        self.cf=cho_factor(matrix,lower=True,overwrite_a=True,check_finite=False)

    def compress_rhs(self,rhs):
        w=rhs*self.physical_rs if rhs.ndim==1 else rhs*self.physical_rs[:,None]
        return np.concatenate((w[:self.physical_split],self.Q.T@w[self.physical_split:]),axis=0)

    def rhs(self,E,B):
        from EMsolver.mas_mixed_mot import coefficients
        return self.compress_rhs(coefficients(self.g,E,B,self.magnetic_weight))

    def coefficients_many(self,E,B):
        e=np.stack((np.sum(E*self.g.solve_tangent1,axis=-1),np.sum(E*self.g.solve_tangent2,axis=-1)),axis=-1).reshape(len(E),-1)
        b=self.magnetic_weight*299792458*np.sum(B*self.g.solve_normals,axis=-1)
        return self.compress_rhs(np.concatenate((e,b),axis=1).T)
