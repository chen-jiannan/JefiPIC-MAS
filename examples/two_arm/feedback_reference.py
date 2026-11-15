'Driven two-arm example: feedback reference. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
from types import SimpleNamespace
import ast,math
import numpy as np
C0=299792458.;EPS0=8.8541878128e-12

def electric_history(points,src,weights,P,K,node,dt,c,eps,history_nodes):
    v=np.asarray(points)[:,None]-np.asarray(src)[None];rr=np.linalg.norm(v,axis=-1);n=v/rr[...,None]
    tau=node-rr/(c*dt);lo=np.floor(tau).astype(int);a=tau-np.floor(tau)
    integer=a<1e-12;lo=np.where(integer,lo-1,lo);a=np.where(integer,1.,a)
    hi=lo+1;oldest=max(0,node-history_nodes+1);valid=(tau>=0)&(lo>=oldest)&(hi<=node)
    j=np.arange(len(src))[None];p0=P[np.clip(lo,0,node),j];p1=P[np.clip(hi,0,node),j];k0=K[np.clip(lo,0,node),j];k1=K[np.clip(hi,0,node),j]
    u=a[...,None];h00=2*u**3-3*u*u+1;h10=u**3-2*u*u+u;h01=-2*u**3+3*u*u;h11=u**3-u*u
    p=h00*p0+dt*h10*k0+h01*p1+dt*h11*k1
    k=((6*u*u-6*u)*p0+(-6*u*u+6*u)*p1)/dt+(3*u*u-4*u+1)*k0+(3*u*u-2*u)*k1
    kd=((12*u-6)*p0+(-12*u+6)*p1)/dt**2+(6*u-4)*k0/dt+(6*u-2)*k1/dt
    for x in (p,k,kd):x*=valid[...,None]
    initial=integer&(lo==-1)&(oldest==0)&(tau>=0)
    p+=P[0][None]*initial[...,None];k+=K[0][None]*initial[...,None]
    T=lambda x:3*n*np.sum(n*x,axis=-1)[...,None]-x
    r=rr[...,None];w=np.asarray(weights)[None,:,None]/(4*np.pi*eps)
    E=np.sum(w*(T(p)/r**3+T(k)/(c*r*r)+(n*np.sum(n*kd,axis=-1)[...,None]-kd)/(c*c*r)),axis=1)
    B=np.sum(w/(c*c)*(np.cross(k,n)/r**2+np.cross(kd,n)/(c*r)),axis=1)
    return E,B

def make_bridge(mode):
    if mode=='SI':L=T=Q=I=E=B=1.;c=C0;eps=EPS0
    else:
        from RBG_Maxwell.Unit_conversion.main import unit_conversion
        ct=unit_conversion('SI_to_LHQCD',coef_J_to_E=1.386275612896618e-15,hbar=1.1822579826363647e-7,c=2.9979090484247153,epsilon0=90631227428685.16)
        L,T,Q,I,E,B=[ct[x] for x in ('meter','second','Coulomb','Ampere','Volt/m','Tesla')];c=2.9979090484247153;eps=90631227428685.16
    return SimpleNamespace(L=L,T=T,Q=Q,I=I,E=E,B=B,c_code=c,epsilon0_code=eps,
        length_si_to_code=lambda x:np.asarray(x)*L,area_si_to_code=lambda x:np.asarray(x)*L**2,
        surface_polarization_density_si_to_code=lambda x:np.asarray(x)*Q/L,
        surface_current_density_si_to_code=lambda x:np.asarray(x)*I/L)

def grid_solver(bridge,shape=(3,3,3),spacing=(.004,.003,.012),origin=(-.004,-.003,-.02),dt_si=1e-9):
    v=.01**3*bridge.L**3
    return SimpleNamespace(x_grid_size_o=shape[0],y_grid_size_o=shape[1],z_grid_size_o=shape[2],total_grid_o=int(np.prod(shape)),
        dx_o=spacing[0]*bridge.L,dy_o=spacing[1]*bridge.L,dz_o=spacing[2]*bridge.L,
        x_left_boundary_o=origin[0]*bridge.L,y_left_boundary_o=origin[1]*bridge.L,z_left_boundary_o=origin[2]*bridge.L,
        dx_s=.01*bridge.L,dy_s=.01*bridge.L,dz_s=.01*bridge.L,c=bridge.c_code,dt=dt_si*bridge.T,t=0.,
        coeff_E=v/(4*np.pi*bridge.epsilon0_code),coeff_B=-v/(4*np.pi*bridge.epsilon0_code*bridge.c_code**2))

def kernel_cpu_dispatch(path):
    cuda=SimpleNamespace(index=0);cuda.grid=lambda dimension:cuda.index
    tree=ast.parse(Path(path).read_text(encoding='utf-8'));wanted={'_hermite_point_state','Jefimenko_point_dipole_kernel_hermite_parallel'};nodes=[]
    for x in tree.body:
        if isinstance(x,ast.FunctionDef) and x.name in wanted:x.decorator_list=[];nodes.append(x)
    if len(nodes)!=2:raise RuntimeError('Native Hermite kernel/reference helper missing')
    env=dict(math=math,cuda=cuda,_truncation_weight=lambda r,cut,width,kind:1.)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    fn=env['Jefimenko_point_dipole_kernel_hermite_parallel']
    class Dispatcher:
        def __getitem__(self,launch):
            def invoke(*args):
                # Argument 21 is total_grid_o in the existing kernel signature.
                total=int(args[21])
                for i in range(total):cuda.index=i;fn(*args)
            return invoke
    return Dispatcher()
