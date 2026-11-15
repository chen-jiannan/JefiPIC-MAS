'Normalized causal electric-dipole mixed PEC boundary operator. Electric and magnetic augmentation is implemented by the two-arm example. See README.md for the executed configurations and validation scopes.'
import numpy as np
from scipy.linalg import cho_factor, cho_solve

EPS0=8.8541878128e-12
C0=299792458.0


def tangents(g):
    return np.stack((g.source_tangent1,g.source_tangent2),axis=1)


def coefficients(g,E,B,magnetic_weight):
    E=np.asarray(E);B=np.asarray(B)
    et=np.stack((np.sum(E*g.solve_tangent1,axis=-1),np.sum(E*g.solve_tangent2,axis=-1)),axis=-1).reshape(-1)
    bn=np.sum(B*g.solve_normals,axis=-1).reshape(-1)
    return np.r_[et,magnetic_weight*C0*bn]


def reference_balance(g,E,B):
    E=np.asarray(E);B=np.asarray(B)
    et=np.stack((np.sum(E*g.solve_tangent1,axis=-1),np.sum(E*g.solve_tangent2,axis=-1)),axis=-1)
    en=np.linalg.norm(et)
    normal=np.linalg.norm(np.sum(B*g.solve_normals,axis=-1))
    full=np.linalg.norm(B)
    # A symmetric source can have identically zero physical Bn but nonzero B.
    # Avoid normalizing mesh/roundoff noise to unit amplitude. This decision is
    # made on the fixed reference record, never during time advancement.
    bn=C0*(full if normal<.01*full else normal)
    if not np.isfinite(en+bn) or min(en,bn)<=1e-300:
        raise ValueError('A fixed nonzero E/B reference is required; provide an explicit magnetic_weight otherwise.')
    return float(en/bn)


def field_blocks(points,g,dt,chunk=16,theta=.5):
    """Independent full-vector E/B maps of [Pprev,Kprev,Know] at a time node."""
    points=np.asarray(points); ns=2*len(g.ant_src_points)
    E=[np.empty((3*len(points),ns)) for _ in range(3)]
    B=[np.empty_like(E[0]) for _ in range(3)]
    st=tangents(g)
    for i in range(0,len(points),chunk):
        x=points[i:i+chunk];v=x[:,None,:]-g.ant_src_points[None,:,:]
        r=np.linalg.norm(v,axis=-1)
        if np.any(r<=0):raise ValueError('Coincident source and observation')
        alpha=1-r/(C0*dt)
        if np.any(alpha<0) or np.any(alpha>1):raise ValueError('Retarded delay is outside one-step history')
        n=v/r[...,None];I=np.eye(3)[None,None,:,:]
        nn=n[..., :,None]*n[...,None,:]
        w=g.source_weights[None,:,None,None]/(4*np.pi*EPS0)
        gp=w*(3*nn-I)/r[...,None,None]**3
        gk=w*(3*nn-I)/(C0*r[...,None,None]**2)
        gd=w*(nn-I)/(C0**2*r[...,None,None])
        aa=alpha[...,None,None]
        f0=(2*theta-1)*aa**3+(1-3*theta)*aa**2+aa
        f1=(1-2*theta)*aa**3+(3*theta-1)*aa**2
        d0=3*(2*theta-1)*aa**2+2*(1-3*theta)*aa+1
        d1=1-d0;dd0=6*(2*theta-1)*aa+2*(1-3*theta)
        tensors=(gp,gp*dt*f0+gk*d0+gd*dd0/dt,gp*dt*f1+gk*d1-gd*dd0/dt)
        sl=slice(3*i,3*(i+len(x)))
        for out,T in zip(E,tensors):
            out[sl]=np.einsum('mjab,jvb->majv',T,st,optimize=True).reshape(3*len(x),ns)
        cross=np.cross(st[None,:,:,:],n[:,:,None,:])
        b0=cross*g.source_weights[None,:,None,None]/(4*np.pi*EPS0*C0**2*r[:,:,None,None]**2)
        b1=b0*r[:,:,None,None]/C0
        B[0][sl]=0
        B[1][sl]=np.transpose(b0*d0+b1*dd0/dt,(0,3,1,2)).reshape(3*len(x),ns)
        B[2][sl]=np.transpose(b0*d1-b1*dd0/dt,(0,3,1,2)).reshape(3*len(x),ns)
    return E,B


class NormalizedMixedMOT:
    def __init__(self,g,dt,reference_E=None,reference_B=None,*,magnetic_weight=None,ridge_rel=1e-4,chunk=16,theta=.5):
        self.g=g;self.dt=float(dt);self.ridge_rel=float(ridge_rel);self.theta=float(theta)
        self.magnetic_weight=reference_balance(g,reference_E,reference_B) if magnetic_weight is None else float(magnetic_weight)
        if not np.isfinite(self.magnetic_weight) or self.magnetic_weight<=0:raise ValueError('Invalid magnetic_weight')
        m=len(g.solve_obs_points);n=2*len(g.ant_src_points)
        self.Hp=np.zeros((3*m,n));self.Hk=np.empty_like(self.Hp);now=np.empty_like(self.Hp)
        # Assemble in chunks to avoid storing all six full-vector surface maps.
        for i in range(0,m,chunk):
            end=min(i+chunk,m);E,B=field_blocks(g.solve_obs_points[i:end],g,self.dt,chunk,theta=self.theta)
            ot=np.stack((g.solve_tangent1[i:end],g.solve_tangent2[i:end]),axis=1)
            for target,ee,bb in zip((self.Hp,self.Hk,now),E,B):
                ev=ee.reshape(end-i,3,n);bv=bb.reshape(end-i,3,n)
                target[2*i:2*end]=np.einsum('mua,man->mun',ot,ev).reshape(2*(end-i),n)
                target[2*m+i:2*m+end]=self.magnetic_weight*C0*np.einsum('ma,man->mn',g.solve_normals[i:end],bv)
        area=np.sqrt(g.solve_weights/np.mean(g.solve_weights))
        self.rs=np.r_[np.repeat(area,2),area]
        now*=self.rs[:,None]
        self.cs=np.linalg.norm(now,axis=0);self.cs=np.where(self.cs>1e-300,self.cs,1.)
        now/=self.cs[None,:];self.As=now
        rng=np.random.default_rng(1234);v=rng.normal(size=n);v/=np.linalg.norm(v)
        for _ in range(8):
            v=now.T@(now@v);v/=max(np.linalg.norm(v),1e-300)
        self.ridge_alpha=self.ridge_rel*np.linalg.norm(now@v)
        H=now.T@now;H.flat[::n+1]+=self.ridge_alpha**2
        self.cf=cho_factor(H,lower=True,overwrite_a=True,check_finite=False)

    def rhs(self,E,B):
        return coefficients(self.g,E,B,self.magnetic_weight)

    def solve(self,b):
        b=np.asarray(b)
        if np.iscomplexobj(b):
            return self.solve(b.real)+1j*self.solve(b.imag)
        weighted=b*self.rs if b.ndim==1 else b*self.rs[:,None]
        return cho_solve(self.cf,self.As.T@weighted,check_finite=False)/(self.cs if b.ndim==1 else self.cs[:,None])

    def step(self,p,k,incident_E,incident_B):
        kn=self.solve(-self.rhs(incident_E,incident_B)-self.Hp@p-self.Hk@k)
        pn=p+self.dt*((1-self.theta)*k+self.theta*kn)
        if not np.all(np.isfinite(kn)) or not np.all(np.isfinite(pn)):
            raise FloatingPointError('Mixed MOT state became nonfinite')
        return pn,kn

    def periodic_response(self,rhs,omega,start=None,rtol=1e-8):
        from scipy.linalg import solve
        zi=np.exp(-1j*omega*self.dt)
        p_over_k=self.dt*((1-self.theta)*zi+self.theta)/(1-zi)
        def mv(q):return q-self.solve(-zi*(p_over_k*(self.Hp@q)+self.Hk@q))
        b=self.solve(-rhs)
        history=zi*(p_over_k*self.Hp+self.Hk)
        history*=self.rs[:,None];history/=self.cs[None,:]
        matrix=self.As.T@(self.As+history)
        matrix.flat[::len(matrix)+1]+=self.ridge_alpha**2
        y=solve(matrix,-self.As.T@(self.rs*rhs),check_finite=False)
        q=y/self.cs
        return q,p_over_k,dict(periodic_solver='direct regularized recurrence, no transient truncation',fixed_point_residual=float(np.linalg.norm(mv(q)-b)/np.linalg.norm(b)))
